#define SDL_MAIN_HANDLED
#include <SDL.h>
#include <SDL_vulkan.h>
#include "prx/libSceAgcDriver/Execution/include/CaptureTrace.hpp"
#include <filesystem>
#include <map>
#include <sstream>
#include "Recompiler.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Texture.hpp"
#include "prx/libSceAgcDriver/Graphics/include/TextureFormat.hpp"
#include "prx/libSceAgcDriver/Graphics/include/DepthSurface.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTarget.hpp"
#include "prx/libSceAgcDriver/Graphics/include/MultisampleTextureLayout.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Resources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/ShaderResources.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Recorder.hpp"
#include "prx/libSceAgcDriver/Graphics/include/Draw.hpp"
#include "prx/libSceAgcDriver/Graphics/include/GuestBufferMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/DisplayBuffer.hpp"
#include "prx/libSceAgcDriver/Execution/include/GuestMemory.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver.hpp"
#include "prx/libSceAgcDriver/Execution/include/Driver/Driver.hpp"
#include "prx/libSceAgcDriver/Submit/include/Dcb.hpp"
#include "prx/libc/include/GuestArena.hpp"
#include "prx/libc/include/GuestAllocations.hpp"
#include "prx/libc/include/GuestWriteWatch.hpp"
#include "SpirvBackend/SpirvModule.hpp"
#include "VulkanTestDevice.hpp"
#include "FixtureDeviceSelection.hpp"
#include <SDL_loadso.h>
#include <algorithm>
#include <array>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <thread>
#include <vector>
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <sys/mman.h>
#ifdef __linux__
#include <fcntl.h>
#include <unistd.h>
#include <map>
#include <iterator>
#endif
#endif


namespace AgcDriver {
struct VulkanDeviceTestAccess {
    static Graphics::Context ContextOf(const VulkanDevice& device) { return device.graphicsContext(); }
};
}
namespace {
using namespace AgcDriver::Graphics;
#ifdef __linux__
struct SharedBacking {
    int file;
    std::size_t bytes;
    std::uint64_t offset;
};
std::map<std::uintptr_t, SharedBacking> sharedBlocks;

bool ResolveSharedBlock(std::uintptr_t address, std::size_t bytes, int* file, std::uint64_t* offset) {
    if (bytes == 0) return false;
    const auto next = sharedBlocks.upper_bound(address);
    if (next == sharedBlocks.begin()) return false;
    const auto entry = std::prev(next);
    const auto lead = address - entry->first;
    if (lead >= entry->second.bytes || bytes > entry->second.bytes - lead) return false;
    *file = fcntl(entry->second.file, F_DUPFD_CLOEXEC, 0);
    *offset = entry->second.offset + lead;
    return *file >= 0;
}

struct SharedBackingScope {
    SharedBackingScope() { GuestArena::GuestArenaSetSharedBacking_nid_postfix(ResolveSharedBlock); }
    ~SharedBackingScope() { GuestArena::GuestArenaSetSharedBacking_nid_postfix(nullptr); }
};
#endif

class TargetMemory {
public:
    explicit TargetMemory(std::size_t bytes, bool registerAllocation = false) : bytes(bytes), registered(registerAllocation) {
#ifdef _WIN32
        pointer = GuestArena::GuestArenaAllocate_nid_postfix(bytes, 65536u);
        GuestArena::GuestArenaCommit_nid_postfix(pointer, bytes, PAGE_READWRITE, bytes);
#else
#ifdef __linux__
        backing = memfd_create("native-depth-htile-clear", MFD_CLOEXEC | MFD_ALLOW_SEALING);
        Require(backing >= 0, "cannot create the native HTILE backing");
        if (ftruncate(backing, bytes + 65536u) != 0 || fcntl(backing, F_ADD_SEALS, F_SEAL_SHRINK) != 0) {
            close(backing);
            Require(false, "cannot size or seal the native HTILE backing");
        }
        void* raw = mmap(nullptr, bytes + 65536u, PROT_READ | PROT_WRITE, MAP_SHARED, backing, 0);
        if (raw == MAP_FAILED) close(backing);
#else
        void* raw = mmap(nullptr, bytes + 65536u, PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
#endif
        Require(raw != MAP_FAILED, "cannot allocate multisample target fixture");
        const auto begin = reinterpret_cast<std::uintptr_t>(raw);
        const auto aligned = (begin + 65535u) & ~std::uintptr_t{65535u};
        if (aligned != begin) munmap(raw, aligned - begin);
        if (aligned + bytes != begin + bytes + 65536u) munmap(reinterpret_cast<void*>(aligned + bytes), begin + 65536u - aligned);
        pointer = reinterpret_cast<void*>(aligned);
#ifdef __linux__
        sharedBlocks.emplace(aligned, SharedBacking{backing, bytes, aligned - begin});
#endif
        GuestWriteWatch::GuestWriteWatchRegister_nid_postfix(pointer, bytes);
#endif
        Require(pointer != nullptr, "multisample target fixture allocation failed");
        std::memset(pointer, 0xa5, bytes);
        if (registered) GuestAllocations::Mutation().Add(pointer, bytes, true, true, true);
    }
    ~TargetMemory() {
        if (registered) GuestAllocations::Mutation().Remove(pointer);
#ifdef _WIN32
        GuestArena::GuestArenaReset_nid_postfix(pointer, bytes);
        GuestArena::GuestArenaRelease_nid_postfix(pointer, bytes);
#else
        GuestWriteWatch::GuestWriteWatchUnregister_nid_postfix(pointer, bytes);
#ifdef __linux__
        sharedBlocks.erase(reinterpret_cast<std::uintptr_t>(pointer));
        close(backing);
#endif
        munmap(pointer, bytes);
#endif
    }
    std::byte* Data() const { return static_cast<std::byte*>(pointer); }
    std::uint64_t Address() const { return reinterpret_cast<std::uint64_t>(pointer); }
private:
    void* pointer = nullptr;
    std::size_t bytes;
    bool registered = false;
#ifdef __linux__
    int backing = -1;
#endif
};

}

namespace {
struct Window {
    SDL_Window* window;
    std::vector<const char*> extensions;
    AgcDriver::PresentationWindow descriptor{};
    explicit Window(SDL_Window* w) : window(w) {
        unsigned count = 0;
        Require(SDL_Vulkan_GetInstanceExtensions(window, &count, nullptr), "SDL extensions count");
        extensions.resize(count);
        Require(SDL_Vulkan_GetInstanceExtensions(window, &count, extensions.data()), "SDL extensions list");
        descriptor = {this, extensions, [](void* p, VkInstance instance) {
            VkSurfaceKHR surface{};
            Require(SDL_Vulkan_CreateSurface(static_cast<Window*>(p)->window, instance, &surface), "SDL surface");
            return surface;
        }, [](void* p, std::uint32_t* w, std::uint32_t* h) {
            int x=0,y=0; SDL_Vulkan_GetDrawableSize(static_cast<Window*>(p)->window, &x, &y);
            *w=static_cast<std::uint32_t>(x); *h=static_cast<std::uint32_t>(y);
        }, 128, 128, {}};
    }
};
void Env(const char* name, const std::string& value) {
#ifdef _WIN32
    Require(_putenv_s(name, value.c_str()) == 0, "set environment");
#else
    Require(setenv(name, value.c_str(), 1) == 0, "set environment");
#endif
}
}
using Fields = std::map<std::string, std::string>;
Fields FieldsOf(const std::string& line) {
    std::istringstream in(line);
    std::string token; Fields fields;
    while (in >> token) {
        const auto eq=token.find('=');
        if (eq!=std::string::npos) fields[token.substr(0,eq)]=token.substr(eq+1);
    }
    return fields;
}
int main(int argc, char** argv) {
    try {
        const bool off = argc == 2 && std::string_view(argv[1]) == "--off";
        const bool direct = argc == 2 && std::string_view(argv[1]) == "--direct";
        Require(argc == 1 || off || direct, "expected --off, --direct or no argument");
        auto temp=std::filesystem::absolute(std::filesystem::current_path() / "build/present-source-readback-tests" / ("run-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count())));
        std::filesystem::create_directories(temp); std::filesystem::current_path(temp);
        const auto trace=(temp/"trace.log").string();
        std::printf("source readback trace: %s\n", trace.c_str()); std::fflush(stdout);
#ifdef _WIN32
        _putenv_s("APS5_NO_GPU_DUMP", ""); _putenv_s("APS5_NO_RESIDENT_PRESENT", "");
        if (off) _putenv_s("APS5_DUMP_PRESENT_SOURCE", "");
#else
        unsetenv("APS5_NO_GPU_DUMP"); unsetenv("APS5_NO_RESIDENT_PRESENT");
        if (off) unsetenv("APS5_DUMP_PRESENT_SOURCE");
#endif
        Env("APS5_DUMP_FRAMES", "4"); Env("APS5_DUMP_FRAMES_EVERY", "1");
        if (!off) Env("APS5_DUMP_PRESENT_SOURCE", "1");
        Env("APS5_CAPTURE_TRACE", "1"); Env("APS5_CAPTURE_TRACE_PATH", trace);
        if (SDL_Init(SDL_INIT_VIDEO) != 0) {
            std::fprintf(stderr,"SDL video unavailable: %s\n", SDL_GetError()); return std::getenv("ANYPS5_REQUIRE_VULKAN") ? 1 : 77;
        }
        SDL_Window* raw=SDL_CreateWindow("Source readback regression", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, 128,128,SDL_WINDOW_VULKAN|SDL_WINDOW_HIDDEN);
        if (!raw) { std::fprintf(stderr,"SDL Vulkan window unavailable: %s\n",SDL_GetError()); SDL_Quit(); return std::getenv("ANYPS5_REQUIRE_VULKAN") ? 1 : 77; }
        {
#ifdef __linux__
            SharedBackingScope sharedBacking;
#endif
            Window window(raw);
            AgcDriver::VulkanDevice device(&window.descriptor);
            auto context=AgcDriver::VulkanDeviceTestAccess::ContextOf(device);
            TargetMemory memory(0x100000u, true);
            struct Cleanup { AgcDriver::VulkanDevice& d; ~Cleanup() { d.WaitIdle(); ClearCachedTextures(d.Device()); } } cleanup{device};
            std::memset(memory.Data(),0,2*65536);
            std::unique_lock lock(AgcDriver::GuestMemory::GpuMutex());
            Require(HostImportFor(context,memory.Address(),0x100000u)!=nullptr,"host import unavailable");
            Require(AgcDriver::GuestMemory::Watched(memory.Address(),2*65536),"write watch unavailable");
            Require(AgcDriver::GuestMemory::CollectWritesUncached(memory.Address(),2*65536)!=0,"guest generation unavailable");
            GuestTextureResource desc{};
            desc.baseAddress=memory.Address(); desc.width=desc.height=128; desc.depthOrLastArray=1; desc.mipCount=1;
            desc.tileMode=TextureTileMode::kR64KBX; desc.dimension=TextureDimension::k2DArray; desc.format=56;
            desc.dstSelX=4; desc.dstSelY=5; desc.dstSelZ=6; desc.dstSelW=7;
            const auto geometry=DescribeSurface(desc);
            Require(geometry.guestBytes==2*65536 && geometry.layerBytes==65536 && geometry.imageLayers==2,"expected two 64 KiB layers");
            auto image=CachedStorageLayers(context,desc,0,2);
            Require(image && image->ImageFormat()==VK_FORMAT_R8G8B8A8_UNORM,"source must be RGBA8");
            for (unsigned i=0;i<4;++i) {
                const std::uint32_t layer=i==3 ? 1u : 0u;
                auto commands=context.recorder->Commands(); context.recorder->Keep(image);
                RecordMemoryBarrier(context,commands,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);
                VkClearColorValue color{}; color.float32[3]=1.f;
                if (i==0 || i==3) color.float32[0]=1.f;
                const VkImageSubresourceRange range{VK_IMAGE_ASPECT_COLOR_BIT,0,1,layer,1};
                context.Function<PFN_vkCmdClearColorImage>("vkCmdClearColorImage")(commands,image->Image(),VK_IMAGE_LAYOUT_GENERAL,&color,1,&range);
                if (i==2) {
                    RecordMemoryBarrier(context,commands,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_TRANSFER_WRITE_BIT);
                    auto pixel=std::make_shared<Buffer>(context,4,VK_BUFFER_USAGE_TRANSFER_SRC_BIT);
                    const std::array<std::byte,4> red{std::byte{255},std::byte{0},std::byte{0},std::byte{255}};
                    std::memcpy(pixel->Bytes().data(),red.data(),red.size());
                    context.recorder->Keep(pixel);
                    VkBufferImageCopy region{}; region.imageSubresource={VK_IMAGE_ASPECT_COLOR_BIT,0,layer,1};
                    region.imageOffset={17,19,0}; region.imageExtent={1,1,1};
                    context.Function<PFN_vkCmdCopyBufferToImage>("vkCmdCopyBufferToImage")(commands,pixel->Handle(),image->Image(),VK_IMAGE_LAYOUT_GENERAL,1,&region);
                }
                RecordMemoryBarrier(context,commands,VK_PIPELINE_STAGE_TRANSFER_BIT,VK_PIPELINE_STAGE_ALL_COMMANDS_BIT,VK_ACCESS_TRANSFER_WRITE_BIT,VK_ACCESS_MEMORY_READ_BIT|VK_ACCESS_MEMORY_WRITE_BIT);
                image->MarkSliceDirty(layer);
                const AgcDriver::DisplayBuffer display{memory.Address()+layer*65536u,direct ? 0x8000000022000000ull : 0x8000000000000000ull,128,128};
                device.Resize(128, 128);
                device.SelectDynamicRange(&display);
                lock.unlock();
                Require(device.AcquireImage(), "acquire was dropped");
                lock.lock();
                Require(device.PresentDisplayBuffer(display), "present was dropped");
                lock.unlock();
                device.QueuePresent();
                device.FinishPresent();
                lock.lock();
            }
            lock.unlock();
            const auto deadline=std::chrono::steady_clock::now()+std::chrono::seconds(5);
            std::string data;
            auto completedAt=std::chrono::steady_clock::time_point{};
            const auto count = [](const std::string& text, const std::string& needle) {
                std::size_t total=0, position=0;
                while ((position=text.find(needle,position))!=std::string::npos) { ++total; position+=needle.size(); }
                return total;
            };
            do {
                std::ifstream f(trace); data.assign(std::istreambuf_iterator<char>(f),std::istreambuf_iterator<char>());
                const bool complete=count(data," present dump=")>=4 && count(data," blit dump=")>=4 && (off || count(data," present-source dump=")>=4);
                if (complete && completedAt==std::chrono::steady_clock::time_point{}) completedAt=std::chrono::steady_clock::now();
                if (complete && (!off || std::chrono::steady_clock::now()-completedAt>=std::chrono::seconds(1))) break;
                std::this_thread::sleep_for(std::chrono::milliseconds(10));
            } while (std::chrono::steady_clock::now()<deadline);
            if (off) Require(data.find(" present-source") == std::string::npos, "disabled source diagnostic emitted an event");
            std::map<int,Fields> presents, sources, blits;
            std::istringstream lines(data); std::string line;
            while (std::getline(lines,line)) {
                auto fields=FieldsOf(line);
                if (!fields.contains("dump")) continue;
                const int index=std::stoi(fields.at("dump"));
                if (index<0 || index>3) continue;
                if (line.find(" present-source ")!=std::string::npos) Require(sources.emplace(index,fields).second,"duplicate source event");
                else if (line.find(" present ")!=std::string::npos) Require(presents.emplace(index,fields).second,"duplicate present event");
                else if (line.find(" blit ")!=std::string::npos) Require(blits.emplace(index,fields).second,"duplicate blit event");
            }
            Require(presents.size()==4 && blits.size()==4,"four normal present submissions required");
            Require(off ? sources.empty() : sources.size()==4,"wrong source event count");
            const std::array<std::string,4> counts{"16384","0","1","16384"};
            for (int i=0;i<4;++i) {
                const auto& present=presents.at(i);
                Require(present.at("resident")=="1", "fixture did not use a resident image");
                if (off) continue;
                Require(present.at("convert")== (direct ? "0" : "1"), "fixture used the wrong resident conversion path");
                Require(blits.at(i).at("direct")== (direct ? "1" : "0"), "fixture used the wrong blit path");
                const auto layer=i==3 ? "1" : "0";
                Require(present.at("selected_layer")==layer,"wrong selected array layer");
                const auto& source=sources.at(i);
                Require(source.at("device")==present.at("device") && source.at("vk_image")==present.at("vk_image") && source.at("storage_object")==present.at("storage_object"),"source and present identity differ");
                Require(source.at("layer")==layer && source.at("vk_format")==present.at("vk_format"),"source and present subresource/format differ");
                Require(source.at("width")=="128" && source.at("height")=="128" && source.at("pixels")=="16384" && source.at("rgb_nonzero")==counts[i],"wrong full source RGB count");
            }
        }
        SDL_DestroyWindow(raw); SDL_Quit();
        for (int i=0; i<4; ++i) {
            char name[32]; std::snprintf(name,sizeof(name),"frame_%03d.bmp",i);
            Require(std::filesystem::exists(temp/name), "normal BMP output missing");
            std::snprintf(name,sizeof(name),"source_%03d.bmp",i);
            if (off) {
                Require(!std::filesystem::exists(temp/name), "disabled source diagnostic wrote a BMP");
                continue;
            }
            std::ifstream source(temp/name, std::ios::binary);
            const std::vector<unsigned char> bmp{std::istreambuf_iterator<char>(source),std::istreambuf_iterator<char>()};
            Require(bmp.size()==54u+128u*128u*4u && bmp[0]=='B' && bmp[1]=='M', "source BMP missing or malformed");
            const auto red = [](const auto& data, std::size_t offset) { return data[offset]==0 && data[offset+1]==0 && data[offset+2]==255; };
            const auto black = [](const auto& data, std::size_t offset) { return data[offset]==0 && data[offset+1]==0 && data[offset+2]==0; };
            Require((i==0 || i==3) ? red(bmp,54) : black(bmp,54), "source BMP first pixel has the wrong color");
            if (i==2) Require(red(bmp,54+(19u*128u+17u)*4u), "source BMP omitted the single red pixel");
            std::snprintf(name,sizeof(name),"frame_%03d.bmp",i);
            std::ifstream output(temp/name, std::ios::binary);
            const std::vector<unsigned char> frame{std::istreambuf_iterator<char>(output),std::istreambuf_iterator<char>()};
            Require(frame.size()>=58 && frame[0]=='B' && frame[1]=='M', "output BMP missing or malformed");
            Require((i==0 || i==3) ? !black(frame,54) : black(frame,54), "paired output BMP first pixel has the wrong brightness");
        }
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"source readback test failed: %s\n",e.what()); return 1; }
}
