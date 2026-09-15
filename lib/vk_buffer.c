/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <errno.h>
#include <inttypes.h>
#include <string.h>

#include <craw/utils.h>

#include "vk_runtime.h"

int craw_vk_buffer_create(CRAWContext *ctx, CRAWVkBuffer *b, VkDeviceSize size,
                          VkBufferUsageFlags usage, enum CRAWVkMemUsage mem)
{
    int err;

    memset(b, 0, sizeof(*b));

    if (!size)
        return CRAW_ERROR(EINVAL);

    VkBufferCreateInfo info = {
        .sType       = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO,
        .size        = size,
        .usage       = usage,
        .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
    };

    err = craw_vk_alloc_buffer(ctx->allocator, &info, mem, &b->buf, &b->alloc,
                               &b->mapped);
    if (err < 0) {
        craw_log(ctx, CRAW_LOG_ERROR,
                 "Unable to allocate a %" PRIu64 " byte buffer\n",
                 (uint64_t)size);
        return err;
    }

    b->size  = size;
    b->usage = mem;

    if (usage & VK_BUFFER_USAGE_SHADER_DEVICE_ADDRESS_BIT) {
        VkBufferDeviceAddressInfo addr_info = {
            .sType  = VK_STRUCTURE_TYPE_BUFFER_DEVICE_ADDRESS_INFO,
            .buffer = b->buf,
        };
        b->addr = ctx->fn.GetBufferDeviceAddress(ctx->device, &addr_info);
    }

    return 0;
}

void craw_vk_buffer_free(CRAWContext *ctx, CRAWVkBuffer *b)
{
    if (!b || !b->buf)
        return;

    craw_vk_free_buffer(ctx->allocator, b->buf, b->alloc);
    memset(b, 0, sizeof(*b));
}

int craw_vk_buffer_flush(CRAWContext *ctx, CRAWVkBuffer *b)
{
    if (!b->mapped)
        return 0;
    return craw_vk_alloc_flush(ctx->allocator, b->alloc, 0, VK_WHOLE_SIZE);
}

int craw_vk_buffer_invalidate(CRAWContext *ctx, CRAWVkBuffer *b)
{
    if (!b->mapped)
        return 0;
    return craw_vk_alloc_invalidate(ctx->allocator, b->alloc, 0, VK_WHOLE_SIZE);
}

void craw_vk_buffer_barrier(CRAWContext *ctx, VkCommandBuffer cmd, CRAWVkBuffer *b,
                            VkPipelineStageFlags2 src_stage, VkAccessFlags2 src_access,
                            VkPipelineStageFlags2 dst_stage, VkAccessFlags2 dst_access)
{
    VkBufferMemoryBarrier2 bar = {
        .sType               = VK_STRUCTURE_TYPE_BUFFER_MEMORY_BARRIER_2,
        .srcStageMask        = src_stage,
        .srcAccessMask       = src_access,
        .dstStageMask        = dst_stage,
        .dstAccessMask       = dst_access,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .buffer              = b->buf,
        .offset              = 0,
        .size                = VK_WHOLE_SIZE,
    };
    VkDependencyInfo dep = {
        .sType                    = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .bufferMemoryBarrierCount = 1,
        .pBufferMemoryBarriers    = &bar,
    };

    ctx->fn.CmdPipelineBarrier2(cmd, &dep);
}
