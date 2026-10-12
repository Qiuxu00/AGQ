// Grouped 1/2/3/4/6/8-bit canonical Huffman with independently decodable byte-padded blocks.
// Binary stdin/stdout avoids platform path encodings. CPU storage tool only.
#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <iterator>
#include <queue>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>
#ifdef _WIN32
#include <fcntl.h>
#include <io.h>
#endif
using Bytes = std::vector<uint8_t>;
static void require(bool yes,const char* text) { if(!yes) throw std::runtime_error(text); }
static void put(Bytes& out,uint64_t v,int bytes) { for(int i=0;i<bytes;++i) out.push_back(uint8_t(v>>(8*i))); }
static uint64_t get(const Bytes& in,size_t& p,int bytes) {
    require(p+bytes<=in.size(),"truncated header"); uint64_t v=0;
    for(int i=0;i<bytes;++i) v|=uint64_t(in[p++])<<(8*i);
    return v;
}
struct Book { std::array<int,256> lengths{}; std::array<uint32_t,256> reverse{}; };
static Book book_from_lengths(const std::array<int,256>& lengths) {
    Book b; b.lengths=lengths; std::vector<std::pair<int,int>> ordered;
    for(int s=0;s<256;++s) if(lengths[s]) { require(lengths[s]<=24,"invalid code length"); ordered.emplace_back(lengths[s],s); }
    require(!ordered.empty(),"empty dictionary"); std::sort(ordered.begin(),ordered.end());
    uint32_t code=0; int previous=0;
    for(auto [width,symbol]:ordered) {
        code<<=(width-previous); require(code < (1u<<width),"invalid prefix lengths");
        uint32_t reversed=0; for(int i=0;i<width;++i) reversed|=((code>>i)&1)<<(width-1-i);
        b.reverse[symbol]=reversed; ++code; previous=width;
    }
    return b;
}
static Book fit(const Bytes& input,std::array<uint64_t,256>& counts,int width) {
    uint64_t buffer=0; int available=0;
    for(auto v:input) {
        buffer|=uint64_t(v)<<available; available+=8;
        while(available>=width) { ++counts[buffer&((1u<<width)-1)]; buffer>>=width; available-=width; }
    }
    require(available==0,"unaligned raw input");
    struct Tree { int left, right, symbol; };
    using Node=std::pair<uint64_t,int>;
    std::priority_queue<Node,std::vector<Node>,std::greater<Node>> heap;
    std::vector<Tree> tree;
    std::array<int,256> lengths{};
    for(int s=0;s<256;++s) if(counts[s]) {
        heap.emplace(counts[s],int(tree.size())); tree.push_back({-1,-1,s});
    }
    require(!heap.empty(),"empty input");
    while(heap.size()>1) {
        auto [a,ai]=heap.top();heap.pop();auto [b,bi]=heap.top();heap.pop();
        heap.emplace(a+b,int(tree.size()));tree.push_back({ai,bi,-1});
    }
    std::vector<std::pair<int,int>> pending{{heap.top().second,0}};
    while(!pending.empty()) {
        auto [index,depth]=pending.back();pending.pop_back();auto node=tree[index];
        if(node.symbol>=0) lengths[node.symbol]=std::max(1,depth);
        else {pending.emplace_back(node.left,depth+1);pending.emplace_back(node.right,depth+1);}
    }
    return book_from_lengths(lengths);
}
static Bytes encode(const Bytes& input,uint32_t block_bytes,int width) {
    require(block_bytes>0 && block_bytes<=1048576,"invalid block size");
    require((width==1 || width==2 || width==3 || width==4 || width==6 || width==8),"unsupported scalar width");
    if(width==3 || width==6) block_bytes-=block_bytes%3;
    require(block_bytes>0,"empty aligned block");
    std::array<uint64_t,256> counts{}; Book b=fit(input,counts,width);
    uint64_t chunks=(input.size()+block_bytes-1)/block_bytes;
    std::vector<uint64_t> offsets{0}; Bytes stream;
    for(size_t start=0;start<input.size();start+=block_bytes) {
        uint64_t buffer=0; int bits=0; size_t end=std::min(input.size(),start+block_bytes);
        uint64_t raw_buffer=0; int raw_bits=0;
        for(size_t i=start;i<end;++i) {
            raw_buffer|=uint64_t(input[i])<<raw_bits; raw_bits+=8;
            while(raw_bits>=width) {
                int symbol=raw_buffer&((1u<<width)-1); raw_buffer>>=width; raw_bits-=width;
                buffer|=uint64_t(b.reverse[symbol])<<bits; bits+=b.lengths[symbol];
                while(bits>=8) { stream.push_back(uint8_t(buffer)); buffer>>=8; bits-=8; }
            }
        }
        require(raw_bits==0,"unaligned block");
        if(bits) stream.push_back(uint8_t(buffer));
        offsets.push_back(stream.size());
    }
    Bytes out{'G','H','C','1'}; put(out,block_bytes,4); put(out,input.size(),8); put(out,chunks,8); put(out,width,1);
    int alphabet=std::max(16,1<<width);
    for(int i=0;i<alphabet;++i) out.push_back(uint8_t(b.lengths[i]));
    for(auto offset:offsets) put(out,offset,8);
    out.insert(out.end(),stream.begin(),stream.end());
    return out;
}
static Bytes decode(const Bytes& input) {
    require(input.size()>=49 && std::string(input.begin(),input.begin()+4)=="GHC1","invalid magic");
    size_t pos=4; uint64_t block=get(input,pos,4),size=get(input,pos,8),chunks=get(input,pos,8);
    require(block>0 && block<=1048576 && size>0 && size<=uint64_t(1)<<32,"invalid dimensions");
    require(chunks==(size+block-1)/block,"invalid chunk count");
    int scalar_width=int(get(input,pos,1));
    require((scalar_width==1 || scalar_width==2 || scalar_width==3 || scalar_width==4 || scalar_width==6 || scalar_width==8) && (block*8)%scalar_width==0 && (size*8)%scalar_width==0,"invalid scalar width/alignment");
    std::array<int,256> lengths{}; for(int i=0;i<std::max(16,1<<scalar_width);++i) lengths[i]=int(get(input,pos,1));
    Book b=book_from_lengths(lengths); int maxwidth=*std::max_element(lengths.begin(),lengths.end());
    std::vector<std::pair<uint8_t,uint8_t>> table(1u<<maxwidth,{0,0});
    for(int s=0;s<256;++s) if(lengths[s]) {
        for(uint32_t index=b.reverse[s];index<table.size();index+=(1u<<lengths[s])) {
            require(table[index].second==0,"overlapping codewords"); table[index]={uint8_t(s),uint8_t(lengths[s])};
        }
    }
    std::vector<uint64_t> offsets; for(uint64_t i=0;i<=chunks;++i) offsets.push_back(get(input,pos,8));
    require(offsets[0]==0 && offsets.back()==input.size()-pos,"invalid offsets");
    Bytes out; out.reserve(size); uint32_t mask=(1u<<maxwidth)-1;
    for(uint64_t chunk=0;chunk<chunks;++chunk) {
        require(offsets[chunk]<offsets[chunk+1] && offsets[chunk+1]<=input.size()-pos,"invalid offset order");
        size_t cursor=pos+offsets[chunk],end=pos+offsets[chunk+1]; uint64_t buffer=0; int bits=0;
        auto symbol=[&]() -> uint8_t {
            while(bits<maxwidth && cursor<end) { buffer|=uint64_t(input[cursor++])<<bits; bits+=8; }
            auto [value,width]=table[buffer&mask]; require(width>0 && bits>=width,"invalid compressed symbol");
            buffer>>=width; bits-=width; return value;
        };
        uint64_t n=std::min(block,size-chunk*block);
        uint64_t raw_buffer=0; int raw_bits=0;
        for(uint64_t i=0;i<n*8/scalar_width;++i) {
            auto value=symbol(); require(value<(1u<<scalar_width),"symbol exceeds source width");
            raw_buffer|=uint64_t(value)<<raw_bits; raw_bits+=scalar_width;
            while(raw_bits>=8) { out.push_back(uint8_t(raw_buffer)); raw_buffer>>=8; raw_bits-=8; }
        }
        require(raw_bits==0,"invalid restored alignment");
        require(cursor==end && bits<8 && buffer==0,"trailing data or nonzero padding");
    }
    return out;
}
int main(int argc,char** argv) {
#ifdef _WIN32
    _setmode(_fileno(stdin),_O_BINARY); _setmode(_fileno(stdout),_O_BINARY);
#endif
    try {
        require(argc>=2,"usage: block_huffman encode [block_bytes] | decode");
        std::ios::sync_with_stdio(false); std::cin.tie(nullptr);
        Bytes input((std::istreambuf_iterator<char>(std::cin)),std::istreambuf_iterator<char>());
        std::string mode=argv[1]; require(mode=="encode" || mode=="decode","invalid mode");
        Bytes out=mode=="encode" ? encode(input,argc>2?uint32_t(std::stoul(argv[2])):16384,argc>3?std::stoi(argv[3]):4):decode(input);
        std::cout.write(reinterpret_cast<const char*>(out.data()),out.size());
        require(bool(std::cout),"write failed"); return 0;
    } catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
