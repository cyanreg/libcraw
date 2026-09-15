/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

/* The only C++ translation unit in libcraw: it compiles the AMD Vulkan
 * Memory Allocator and exposes the handful of operations the rest of the
 * library needs behind the C interface declared in vk_runtime.h. */

#include <errno.h>
#include <stdlib.h>
#include <string.h>

#define VMA_IMPLEMENTATION
#include "vk_mem_alloc.h"

extern "C" {
#include "vk_runtime.h"
}

struct CRAWVkAllocator {
    VmaAllocator vma;
    CRAWContext *ctx;
};

static inline VmaAllocation craw_vma_alloc(CRAWVkAllocation *a)
{
    return reinterpret_cast<VmaAllocation>(a);
}

static inline CRAWVkAllocation *craw_vma_wrap(VmaAllocation a)
{
    return reinterpret_cast<CRAWVkAllocation *>(a);
}

extern "C" int craw_vk_allocator_create(CRAWContext *ctx, CRAWVkAllocator **out)
{
    VkResult res;
    CRAWVkAllocator *a;

    /* Build flags pass -DVMA_STATIC_VULKAN_FUNCTIONS=0
     * -DVMA_DYNAMIC_VULKAN_FUNCTIONS=1, so the two loaders are enough:
     * VMA imports everything else it needs through them, and the library
     * still never touches a symbol exported by libvulkan. */
    VmaVulkanFunctions vk_fn = {};
    vk_fn.vkGetInstanceProcAddr = ctx->get_proc_addr;
    vk_fn.vkGetDeviceProcAddr   = ctx->fn.GetDeviceProcAddr;

    VmaAllocatorCreateInfo info = {};
    info.flags                = VMA_ALLOCATOR_CREATE_BUFFER_DEVICE_ADDRESS_BIT;
    info.physicalDevice       = ctx->phys_device;
    info.device               = ctx->device;
    info.pAllocationCallbacks = ctx->alloc;
    info.pVulkanFunctions     = &vk_fn;
    info.instance             = ctx->instance;
    info.vulkanApiVersion     = VK_API_VERSION_1_4;

    a = static_cast<CRAWVkAllocator *>(calloc(1, sizeof(*a)));
    if (!a)
        return CRAW_ERROR(ENOMEM);
    a->ctx = ctx;

    res = vmaCreateAllocator(&info, &a->vma);
    if (res != VK_SUCCESS) {
        craw_log(ctx, CRAW_LOG_ERROR, "Unable to create the VMA allocator: %s\n",
                 craw_vk_ret2str(res));
        free(a);
        return craw_vk_ret2err(res);
    }

    *out = a;
    return 0;
}

extern "C" void craw_vk_allocator_destroy(CRAWVkAllocator **alloc)
{
    CRAWVkAllocator *a;

    if (!alloc || !*alloc)
        return;
    a = *alloc;

    if (a->vma)
        vmaDestroyAllocator(a->vma);

    free(a);
    *alloc = NULL;
}

static void craw_vma_flags(enum CRAWVkMemUsage usage, VmaAllocationCreateInfo *ci)
{
    switch (usage) {
    case CRAW_MEM_HOST_UPLOAD:
        ci->usage = VMA_MEMORY_USAGE_AUTO;
        ci->flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT;
        break;
    case CRAW_MEM_HOST_READBACK:
        /* RANDOM also asks VMA for HOST_CACHED, which is what makes a
         * readback buffer usable at all. */
        ci->usage = VMA_MEMORY_USAGE_AUTO;
        ci->flags = VMA_ALLOCATION_CREATE_MAPPED_BIT |
                    VMA_ALLOCATION_CREATE_HOST_ACCESS_RANDOM_BIT;
        break;
    case CRAW_MEM_DEVICE:
    default:
        ci->usage = VMA_MEMORY_USAGE_AUTO_PREFER_DEVICE;
        ci->flags = 0;
        break;
    }
}

extern "C" int craw_vk_alloc_buffer(CRAWVkAllocator *a, const VkBufferCreateInfo *info,
                                    enum CRAWVkMemUsage usage, VkBuffer *buf,
                                    CRAWVkAllocation **alloc, void **mapped)
{
    VkResult res;
    VmaAllocation va = VK_NULL_HANDLE;
    VmaAllocationInfo ai = {};
    VmaAllocationCreateInfo ci = {};

    craw_vma_flags(usage, &ci);

    res = vmaCreateBuffer(a->vma, info, &ci, buf, &va, &ai);
    if (res != VK_SUCCESS) {
        craw_log(a->ctx, CRAW_LOG_ERROR, "Unable to allocate a buffer: %s\n",
                 craw_vk_ret2str(res));
        return craw_vk_ret2err(res);
    }

    *alloc = craw_vma_wrap(va);
    if (mapped)
        *mapped = ai.pMappedData;

    return 0;
}

extern "C" int craw_vk_alloc_image(CRAWVkAllocator *a, const VkImageCreateInfo *info,
                                   VkImage *img, CRAWVkAllocation **alloc)
{
    VkResult res;
    VmaAllocation va = VK_NULL_HANDLE;
    VmaAllocationCreateInfo ci = {};

    craw_vma_flags(CRAW_MEM_DEVICE, &ci);

    res = vmaCreateImage(a->vma, info, &ci, img, &va, NULL);
    if (res != VK_SUCCESS) {
        craw_log(a->ctx, CRAW_LOG_ERROR, "Unable to allocate an image: %s\n",
                 craw_vk_ret2str(res));
        return craw_vk_ret2err(res);
    }

    *alloc = craw_vma_wrap(va);
    return 0;
}

extern "C" void craw_vk_free_buffer(CRAWVkAllocator *a, VkBuffer buf,
                                    CRAWVkAllocation *alloc)
{
    vmaDestroyBuffer(a->vma, buf, craw_vma_alloc(alloc));
}

extern "C" void craw_vk_free_image(CRAWVkAllocator *a, VkImage img,
                                   CRAWVkAllocation *alloc)
{
    vmaDestroyImage(a->vma, img, craw_vma_alloc(alloc));
}

extern "C" int craw_vk_alloc_flush(CRAWVkAllocator *a, CRAWVkAllocation *alloc,
                                   VkDeviceSize offset, VkDeviceSize size)
{
    /* A no-op on coherent memory, which VMA knows and skips. */
    return craw_vk_ret2err(vmaFlushAllocation(a->vma, craw_vma_alloc(alloc),
                                              offset, size));
}

extern "C" int craw_vk_alloc_invalidate(CRAWVkAllocator *a, CRAWVkAllocation *alloc,
                                        VkDeviceSize offset, VkDeviceSize size)
{
    return craw_vk_ret2err(vmaInvalidateAllocation(a->vma, craw_vma_alloc(alloc),
                                                   offset, size));
}
