#pragma once

class ImageBdaAliasPrograms {
public:
    ImageBdaAliasPrograms(const Context& c, std::span<const std::uint32_t> writer, std::span<const std::uint32_t> reader) : c(c), output(c, 4, VK_BUFFER_USAGE_STORAGE_BUFFER_BIT) {
        VkDescriptorSetLayoutBinding bindings[2]{{0, VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}, {1, VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1, VK_SHADER_STAGE_COMPUTE_BIT, nullptr}};
        VkDescriptorSetLayoutCreateInfo layoutInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO};
        layoutInfo.bindingCount = 2;
        layoutInfo.pBindings = bindings;
        Check(c.Function<PFN_vkCreateDescriptorSetLayout>("vkCreateDescriptorSetLayout")(c.device, &layoutInfo, nullptr, &setLayout), "alias set layout");
        const VkPushConstantRange push{VK_SHADER_STAGE_COMPUTE_BIT, 0, 8};
        VkPipelineLayoutCreateInfo layout{VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO};
        layout.setLayoutCount = 1;
        layout.pSetLayouts = &setLayout;
        layout.pushConstantRangeCount = 1;
        layout.pPushConstantRanges = &push;
        Check(c.Function<PFN_vkCreatePipelineLayout>("vkCreatePipelineLayout")(c.device, &layout, nullptr, &pipelineLayout), "alias pipeline layout");
        const VkDescriptorPoolSize sizes[2]{{VK_DESCRIPTOR_TYPE_STORAGE_IMAGE, 1}, {VK_DESCRIPTOR_TYPE_STORAGE_BUFFER, 1}};
        VkDescriptorPoolCreateInfo poolInfo{VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO};
        poolInfo.maxSets = 1;
        poolInfo.poolSizeCount = 2;
        poolInfo.pPoolSizes = sizes;
        Check(c.Function<PFN_vkCreateDescriptorPool>("vkCreateDescriptorPool")(c.device, &poolInfo, nullptr, &pool), "alias pool");
        VkDescriptorSetAllocateInfo allocate{VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO};
        allocate.descriptorPool = pool;
        allocate.descriptorSetCount = 1;
        allocate.pSetLayouts = &setLayout;
        Check(c.Function<PFN_vkAllocateDescriptorSets>("vkAllocateDescriptorSets")(c.device, &allocate, &set), "alias descriptor set");
        const auto make = [&](std::span<const std::uint32_t> words, VkShaderModule& module, VkPipeline& pipeline) {
            VkShaderModuleCreateInfo info{VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO};
            info.codeSize = words.size_bytes();
            info.pCode = words.data();
            Check(c.Function<PFN_vkCreateShaderModule>("vkCreateShaderModule")(c.device, &info, nullptr, &module), "alias shader module");
            VkComputePipelineCreateInfo pipelineInfo{VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO};
            pipelineInfo.stage = {VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO, nullptr, 0, VK_SHADER_STAGE_COMPUTE_BIT, module, "main", nullptr};
            pipelineInfo.layout = pipelineLayout;
            Check(c.Function<PFN_vkCreateComputePipelines>("vkCreateComputePipelines")(c.device, VK_NULL_HANDLE, 1, &pipelineInfo, nullptr, &pipeline), "alias compute pipeline");
        };
        make(writer, modules[0], pipelines[0]);
        make(reader, modules[1], pipelines[1]);
    }
    ~ImageBdaAliasPrograms() {
        for (auto p : pipelines) if (p) c.Function<PFN_vkDestroyPipeline>("vkDestroyPipeline")(c.device, p, nullptr);
        for (auto m : modules) if (m) c.Function<PFN_vkDestroyShaderModule>("vkDestroyShaderModule")(c.device, m, nullptr);
        if (pool) c.Function<PFN_vkDestroyDescriptorPool>("vkDestroyDescriptorPool")(c.device, pool, nullptr);
        if (pipelineLayout) c.Function<PFN_vkDestroyPipelineLayout>("vkDestroyPipelineLayout")(c.device, pipelineLayout, nullptr);
        if (setLayout) c.Function<PFN_vkDestroyDescriptorSetLayout>("vkDestroyDescriptorSetLayout")(c.device, setLayout, nullptr);
    }
    void BindImage(VkImageView view) {
        const VkDescriptorImageInfo image{VK_NULL_HANDLE, view, VK_IMAGE_LAYOUT_GENERAL};
        const VkDescriptorBufferInfo buffer{output.Handle(), 0, 4};
        VkWriteDescriptorSet writes[2]{{VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}, {VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET}};
        writes[0].dstSet = set; writes[0].dstBinding = 0; writes[0].descriptorCount = 1; writes[0].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE; writes[0].pImageInfo = &image;
        writes[1].dstSet = set; writes[1].dstBinding = 1; writes[1].descriptorCount = 1; writes[1].descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER; writes[1].pBufferInfo = &buffer;
        c.Function<PFN_vkUpdateDescriptorSets>("vkUpdateDescriptorSets")(c.device, 2, writes, 0, nullptr);
    }
    void Dispatch(VkCommandBuffer commands, unsigned which, std::uint64_t address = 0) {
        c.Function<PFN_vkCmdBindPipeline>("vkCmdBindPipeline")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelines[which]);
        c.Function<PFN_vkCmdBindDescriptorSets>("vkCmdBindDescriptorSets")(commands, VK_PIPELINE_BIND_POINT_COMPUTE, pipelineLayout, 0, 1, &set, 0, nullptr);
        if (which == 1) c.Function<PFN_vkCmdPushConstants>("vkCmdPushConstants")(commands, pipelineLayout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(address), &address);
        c.Function<PFN_vkCmdDispatch>("vkCmdDispatch")(commands, 1, 1, 1);
    }
    std::uint32_t Result() { std::uint32_t x = 0; std::memcpy(&x, output.Bytes().data(), 4); return x; }
private:
    const Context& c;
    Buffer output;
    VkDescriptorSetLayout setLayout = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout = VK_NULL_HANDLE;
    VkDescriptorPool pool = VK_NULL_HANDLE;
    VkDescriptorSet set = VK_NULL_HANDLE;
    VkShaderModule modules[2]{};
    VkPipeline pipelines[2]{};
};

bool imageBdaAliasTest(const Device& device, Recorder& recorder, std::span<const std::uint32_t> writer, std::span<const std::uint32_t> reader) {
    const auto& base = device.GetContext();
    if (!base.bufferDeviceAddress || base.hostImportAlignment == 0) { std::cout << "SKIP: BDA or host import unavailable\n"; return false; }
    std::lock_guard gpu(AgcDriver::GuestMemory::GpuMutex());
    constexpr std::size_t bytes = 65536;
    void* block = AllocateWatched(bytes, bytes);
    if (block == nullptr) { std::cout << "SKIP: watched allocation unavailable\n"; return false; }
    struct Release { void* p; ~Release() { ReleaseWatched(p, bytes); } } release{block};
    std::memset(block, 0x55, bytes);
    const auto address = reinterpret_cast<std::uint64_t>(block);
    GuestAllocations::Mutation().Add(block, bytes, true, true, true);
    struct Unregister { const Context& c; void* p; std::uint64_t a; ~Unregister() { GuestAllocations::Mutation().Remove(p); HostImportFor(c, a, bytes); } } unregister{base, block, address};
    const auto* initial = HostImportFor(base, address, bytes);
    if (initial == nullptr || initial->buffer == VK_NULL_HANDLE || initial->address == 0) { std::cout << "SKIP: no GPU host import\n"; return false; }
    TextureDetiler detiler(base);
    auto context = base;
    context.detiler = &detiler;
    context.recorder = &recorder;
    GuestTextureResource resource{};
    resource.baseAddress = address; resource.width = 128; resource.height = 128; resource.mipCount = 1;
    resource.tileMode = TextureTileMode::kR64KBX; resource.dimension = TextureDimension::k2D; resource.format = 60;
    resource.dstSelX = 4; resource.dstSelY = 5; resource.dstSelZ = 6; resource.dstSelW = 7;
    Require(DescribeSurface(resource).guestBytes == bytes, "alias image extent mismatch");
    auto image = StorageTexture::Create(context, detiler, resource, 0);
    Require(image->ImageFormat() == VK_FORMAT_R8G8B8A8_UINT, "alias image is not RGBA8_UINT");
    ImageBdaAliasPrograms programs(context, writer, reader);
    GuestBufferMemory memory(context);
    struct Drain { Recorder& recorder; const Device& device; ~Drain() noexcept { try { recorder.Sync(); device.WaitQueue(); } catch (...) { std::terminate(); } } } drain{recorder, device};
    programs.BindImage(image->StorageView(0, false, true));
    {
        const auto commands = recorder.Commands();
        recorder.Keep(image);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT, VK_ACCESS_SHADER_WRITE_BIT);
        programs.Dispatch(commands, 0);
        image->MarkDirty();
    }
    Require(StorageTexture::FindPending(address, bytes) == image, "compute image not pending");
    memory.AcquireRegistered();
    memory.Upload(true);
    Require(memory.HoldsLease(), "addressable build did not hold a lease");
    const bool pendingAfterUpload = StorageTexture::FindPending(address, bytes) == image;
    const auto reads = memory.InPlaceReads();
    const bool inPlace = std::any_of(reads.begin(), reads.end(), [&](const auto& x) { return x.first <= address && x.second >= address + 4; });
    Require(inPlace, "BDA range not served in place from host-backed allocation");
    const auto* actual = ExistingHostImportFor(context, address, 4);
    Require(actual != nullptr && actual->buffer != VK_NULL_HANDLE && actual->address != 0, "host import disappeared");
    const auto ranges = memory.AddressRanges();
    const auto range = std::find_if(ranges.begin(), ranges.end(), [&](const auto& x) { return x.begin <= address && x.end >= address + 4; });
    Require(range != ranges.end(), "BDA address table misses the image");
    const std::uint64_t gpuAddress = range->deviceAddress + address - range->begin;
    Require(gpuAddress == actual->address + address - actual->base, "BDA address does not name the GPU host import");
    {
        const auto commands = recorder.Commands();
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_ALL_COMMANDS_BIT | VK_PIPELINE_STAGE_HOST_BIT, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_ACCESS_SHADER_WRITE_BIT | VK_ACCESS_TRANSFER_WRITE_BIT | VK_ACCESS_HOST_WRITE_BIT, VK_ACCESS_SHADER_READ_BIT);
        programs.Dispatch(commands, 1, gpuAddress);
        RecordMemoryBarrier(context, commands, VK_PIPELINE_STAGE_COMPUTE_SHADER_BIT, VK_PIPELINE_STAGE_HOST_BIT, VK_ACCESS_SHADER_WRITE_BIT, VK_ACCESS_HOST_READ_BIT);
        recorder.Submit();
        device.WaitQueue();
        recorder.Sync();
    }
    const auto got = programs.Result();
    memory.WriteBack();
    StorageTexture::FlushPending(address, bytes, nullptr, "image BDA alias test cleanup");
    recorder.Submit(); device.WaitQueue(); recorder.Sync();
    std::printf("image BDA alias: gpu=0x%08x expected=0x87654321 forced_flush=%d pending_after_upload=%d\n", got, std::getenv("APS5_BDA_FLUSH") != nullptr, pendingAfterUpload);
    Require(got == 0x87654321u, "pending image results not visible to GPU BDA alias");
    return true;
}
