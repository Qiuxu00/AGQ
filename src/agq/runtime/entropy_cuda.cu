#include <torch/extension.h>
#include <ATen/Parallel.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <zlib.h>
#include <atomic>

// Original HBM1/GHC1 independent streams; no change to persisted codec framing.
__global__ void unpack(const uint8_t* data,const int64_t* offsets,const int32_t* root,
                       const int32_t* children,const int32_t* checkpoints,uint8_t* out,int64_t size,int block,int width,int64_t chunks,int segments){
    int64_t work=int64_t(blockIdx.x)*blockDim.x+threadIdx.x;
    if(work>=chunks*segments)return;
    int64_t chunk=work/segments;int segment=work%segments;
    if(checkpoints[work]<0)return;
    int64_t start=offsets[chunk],end=offsets[chunk+1],base=chunk*block;
    int n=int((size-base)<block?(size-base):block),emitted=0,bits=0;
    int count=n*8/width-segment*256;count=count<256?count:256;base+=segment*256*width/8;
    int64_t bitpos=checkpoints[work];unsigned acc=0;
    for(int j=0;j<count;++j){
        int64_t at=start+(bitpos>>3);unsigned word=0;
        if(at<end)word=data[at];
        if(at+1<end)word|=unsigned(data[at+1])<<8;
        if(at+2<end)word|=unsigned(data[at+2])<<16;
        int entry=root[(word>>(bitpos&7))&1023];unsigned symbol;
        if(entry>0){int len=entry>>24;if(bitpos+len>(end-start)*8)return;bitpos+=len;symbol=entry&255;}
        else if(entry<0){
            int node=-entry-1;bitpos+=10;
            while(node>=0){if(bitpos>=(end-start)*8)return;int branch=(data[start+(bitpos>>3)]>>(bitpos&7))&1;++bitpos;node=children[node*2+branch];if(node==-2147483647)return;}
            symbol=unsigned(-node-1);
        }else return;
        acc|=symbol<<bits;bits+=width;
        while(bits>=8){out[base+emitted++]=uint8_t(acc);acc>>=8;bits-=8;}
    }
}
torch::Tensor decode(torch::Tensor data,torch::Tensor offsets,torch::Tensor root,torch::Tensor children,
                     torch::Tensor checkpoints,int64_t size,int64_t block,int64_t width){
    TORCH_CHECK(data.is_cuda()&&offsets.is_cuda()&&root.is_cuda()&&children.is_cuda(),"CUDA buffers required");
    TORCH_CHECK(data.is_contiguous()&&offsets.is_contiguous()&&root.is_contiguous()&&children.is_contiguous(),"contiguous buffers required");
    TORCH_CHECK(data.scalar_type()==torch::kUInt8&&offsets.scalar_type()==torch::kInt64&&root.scalar_type()==torch::kInt32&&children.scalar_type()==torch::kInt32,"invalid dtype");
    TORCH_CHECK(data.device()==offsets.device()&&data.device()==root.device()&&data.device()==children.device(),"device mismatch");
    TORCH_CHECK(size>0&&block>0&&(width==1||width==2||width==3||width==4||width==6||width==8)&&size*8%width==0&&block*8%width==0,"invalid dimensions");
    int64_t chunks=(size+block-1)/block;
    int segments=(block*8/width+255)/256;
    TORCH_CHECK(offsets.numel()==chunks+1&&root.numel()==1024&&children.numel()%2==0,"invalid lookup metadata");
    TORCH_CHECK(checkpoints.is_cuda()&&checkpoints.device()==data.device()&&checkpoints.scalar_type()==torch::kInt32&&checkpoints.is_contiguous()&&checkpoints.numel()==chunks*segments,"invalid segment checkpoints");
    c10::cuda::CUDAGuard guard(data.device());auto out=torch::empty({size},data.options());
    unpack<<<(chunks*segments+127)/128,128,0,at::cuda::getCurrentCUDAStream()>>>(data.data_ptr<uint8_t>(),offsets.data_ptr<int64_t>(),root.data_ptr<int32_t>(),children.data_ptr<int32_t>(),checkpoints.data_ptr<int32_t>(),out.data_ptr<uint8_t>(),size,int(block),int(width),chunks,segments);
    C10_CUDA_KERNEL_LAUNCH_CHECK();return out;
}

torch::Tensor build_checkpoints(torch::Tensor data,torch::Tensor offsets,torch::Tensor root,torch::Tensor children,int64_t size,int64_t block,int64_t width){
    TORCH_CHECK(!data.is_cuda()&&!offsets.is_cuda()&&!root.is_cuda()&&!children.is_cuda(),"CPU index construction required");
    int64_t chunks=(size+block-1)/block,segments=(block*8/width+255)/256;
    auto result=torch::full({chunks,segments},-1,torch::TensorOptions().dtype(torch::kInt32));
    auto src=data.data_ptr<uint8_t>();auto idx=offsets.data_ptr<int64_t>();auto lut=root.data_ptr<int32_t>();auto tree=children.data_ptr<int32_t>();auto dst=result.data_ptr<int32_t>();
    at::parallel_for(0,chunks,1,[&](int64_t begin,int64_t end){
        for(int64_t c=begin;c<end;++c){
            int64_t start=idx[c],limit=idx[c+1],bitpos=0,n=std::min(block,size-c*block)*8/width;
            for(int64_t j=0;j<n;++j){
                if(j%256==0)dst[c*segments+j/256]=int32_t(bitpos);
                int64_t at=start+(bitpos>>3);unsigned word=0;
                if(at<limit)word=src[at];if(at+1<limit)word|=unsigned(src[at+1])<<8;if(at+2<limit)word|=unsigned(src[at+2])<<16;
                int entry=lut[(word>>(bitpos&7))&1023];
                if(entry>0)bitpos+=entry>>24;
                else if(entry<0){int node=-entry-1;bitpos+=10;while(node>=0){TORCH_CHECK(bitpos<(limit-start)*8,"truncated code");int b=(src[start+(bitpos>>3)]>>(bitpos&7))&1;++bitpos;node=tree[node*2+b];TORCH_CHECK(node!=-2147483647,"invalid code");}}
                else TORCH_CHECK(false,"invalid prefix");
                TORCH_CHECK(bitpos<=(limit-start)*8,"truncated stream");
            }
        }
    });return result;
}
torch::Tensor decode_zlib(torch::Tensor data,torch::Tensor offsets,int64_t size,int64_t block){
    TORCH_CHECK(!data.is_cuda()&&!offsets.is_cuda()&&data.is_contiguous()&&offsets.is_contiguous(),"CPU buffers required");
    TORCH_CHECK(data.scalar_type()==torch::kUInt8&&offsets.scalar_type()==torch::kInt64&&size>0&&block==16384,"invalid zlib metadata");
    int64_t chunks=(size+block-1)/block;TORCH_CHECK(offsets.numel()==chunks+1,"invalid zlib offsets");
    auto out=torch::empty({size},torch::TensorOptions().dtype(torch::kUInt8).device(torch::kCPU).pinned_memory(true));
    auto src=data.data_ptr<uint8_t>();auto dst=out.data_ptr<uint8_t>();auto index=offsets.data_ptr<int64_t>();
    TORCH_CHECK(index[0]==0&&index[chunks]==data.numel(),"invalid zlib size");
    at::parallel_for(0,chunks,1,[&](int64_t begin,int64_t end){
        for(int64_t c=begin;c<end;++c){
            TORCH_CHECK(index[c]>=0&&index[c]<index[c+1]&&index[c+1]<=data.numel(),"invalid block");
            uLongf wanted=std::min(block,size-c*block);uLongf actual=wanted;
            int code=uncompress(dst+c*block,&actual,src+index[c],index[c+1]-index[c]);
            TORCH_CHECK(code==Z_OK&&actual==wanted,"zlib decode failed");
        }
    });return out;
}
int cpu_parallel_workers(){std::atomic<unsigned> workers{0};at::parallel_for(0,1024,1,[&](int64_t begin,int64_t end){workers.fetch_or(1u<<at::get_thread_num());});return __builtin_popcount(workers.load());}
PYBIND11_MODULE(TORCH_EXTENSION_NAME,m){m.def("decode",&decode);m.def("decode_zlib",&decode_zlib);m.def("build_checkpoints",&build_checkpoints);m.def("cpu_parallel_workers",&cpu_parallel_workers);}
