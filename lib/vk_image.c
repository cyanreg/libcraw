/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <errno.h>
#include <string.h>

#include <craw/utils.h>

#include "vk_runtime.h"

int craw_vk_image_create(CRAWContext *ctx, CRAWVkImage *img, uint32_t width,
                         uint32_t height, VkFormat format, VkImageUsageFlags usage)
{
    int err;
    VkResult res;

    memset(img, 0, sizeof(*img));

    if (!width || !height)
        return CRAW_ERROR(EINVAL);

    VkImageCreateInfo info = {
        .sType         = VK_STRUCTURE_TYPE_IMAGE_CREATE_INFO,
        .imageType     = VK_IMAGE_TYPE_2D,
        .format        = format,
        .extent        = { width, height, 1 },
        .mipLevels     = 1,
        .arrayLayers   = 1,
        .samples       = VK_SAMPLE_COUNT_1_BIT,
        .tiling        = VK_IMAGE_TILING_OPTIMAL,
        .usage         = usage,
        .sharingMode   = VK_SHARING_MODE_EXCLUSIVE,
        .initialLayout = VK_IMAGE_LAYOUT_UNDEFINED,
    };

    err = craw_vk_alloc_image(ctx->allocator, &info, &img->image, &img->alloc);
    if (err < 0) {
        craw_log(ctx, CRAW_LOG_ERROR,
                 "Unable to allocate a %ux%u image (format %i)\n",
                 width, height, (int)format);
        return err;
    }

    VkImageViewCreateInfo view_info = {
        .sType    = VK_STRUCTURE_TYPE_IMAGE_VIEW_CREATE_INFO,
        .image    = img->image,
        .viewType = VK_IMAGE_VIEW_TYPE_2D,
        .format   = format,
        .subresourceRange = {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        },
    };
    res = ctx->fn.CreateImageView(ctx->device, &view_info, ctx->alloc, &img->view);
    if (res != VK_SUCCESS) {
        craw_log(ctx, CRAW_LOG_ERROR, "Unable to create an image view: %s\n",
                 craw_vk_ret2str(res));
        craw_vk_free_image(ctx->allocator, img->image, img->alloc);
        memset(img, 0, sizeof(*img));
        return craw_vk_ret2err(res);
    }

    img->format = format;
    img->width  = width;
    img->height = height;
    img->layout = VK_IMAGE_LAYOUT_UNDEFINED;
    img->stage  = VK_PIPELINE_STAGE_2_NONE;
    img->access = 0;
    img->owned  = true;

    return 0;
}

void craw_vk_image_wrap(CRAWVkImage *img, const CRAWImage *ext)
{
    *img = (CRAWVkImage) {
        .image  = ext->image,
        .view   = ext->view,
        .format = ext->format,
        .width  = ext->width,
        .height = ext->height,

        /* The caller is responsible for making its own prior work visible,
         * so anything the library adds on top has to assume the worst. */
        .layout = ext->layout,
        .stage  = VK_PIPELINE_STAGE_2_ALL_COMMANDS_BIT,
        .access = VK_ACCESS_2_MEMORY_READ_BIT | VK_ACCESS_2_MEMORY_WRITE_BIT,

        .alloc  = NULL,
        .owned  = false,
    };
}

void craw_vk_image_free(CRAWContext *ctx, CRAWVkImage *img)
{
    if (!img || !img->owned)
        return;

    if (img->view)
        ctx->fn.DestroyImageView(ctx->device, img->view, ctx->alloc);
    if (img->image)
        craw_vk_free_image(ctx->allocator, img->image, img->alloc);

    memset(img, 0, sizeof(*img));
}

void craw_vk_image_barrier(CRAWContext *ctx, VkCommandBuffer cmd, CRAWVkImage *img,
                           VkImageLayout new_layout, VkPipelineStageFlags2 dst_stage,
                           VkAccessFlags2 dst_access)
{
    if (img->layout == new_layout && img->stage == dst_stage &&
        img->access == dst_access)
        return;

    VkImageMemoryBarrier2 bar = {
        .sType               = VK_STRUCTURE_TYPE_IMAGE_MEMORY_BARRIER_2,
        .srcStageMask        = img->stage,
        .srcAccessMask       = img->access,
        .dstStageMask        = dst_stage,
        .dstAccessMask       = dst_access,
        .oldLayout           = img->layout,
        .newLayout           = new_layout,
        .srcQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .dstQueueFamilyIndex = VK_QUEUE_FAMILY_IGNORED,
        .image               = img->image,
        .subresourceRange = {
            .aspectMask     = VK_IMAGE_ASPECT_COLOR_BIT,
            .baseMipLevel   = 0,
            .levelCount     = 1,
            .baseArrayLayer = 0,
            .layerCount     = 1,
        },
    };
    VkDependencyInfo dep = {
        .sType                   = VK_STRUCTURE_TYPE_DEPENDENCY_INFO,
        .imageMemoryBarrierCount = 1,
        .pImageMemoryBarriers    = &bar,
    };

    ctx->fn.CmdPipelineBarrier2(cmd, &dep);

    img->layout = new_layout;
    img->stage  = dst_stage;
    img->access = dst_access;
}
