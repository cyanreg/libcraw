/* Copyright © 2026, Lynne. BSD-2-Clause, see LICENSE.md. */

#include <errno.h>
#include <stdio.h>
#include <string.h>

#include <craw/utils.h>

#include "vk_runtime.h"

/* All descriptors live in set 0 and are pushed, so these bounds only have
 * to cover the widest stage the pipeline has. */
#define CRAW_MAX_BINDINGS     16
#define CRAW_MAX_SPEC_ENTRIES 16
#define CRAW_MAX_SPEC_DATA    256
/* DESIGN.md caps push constants here; every Vulkan 1.4 device allows at
 * least this much, so a stage that fits is portable by construction. */
#define CRAW_MAX_PUSH_SIZE    128

/* DESIGN.md: the workgroup size is always specialization constants
 * 253/254/255, the FFmpeg convention, leaving everything below to shaders. */
#define CRAW_SPEC_ID_WG_X 253

static inline size_t craw_align4(size_t v)
{
    return (v + 3) & ~(size_t)3;
}

int craw_vk_shader_create(CRAWContext *ctx, CRAWVkShader *sh,
                          const CRAWVkShaderDesc *desc)
{
    const CRAWVkFunctions *fn = &ctx->fn;
    VkResult res;
    int err;
    VkDescriptorSetLayoutBinding bindings[CRAW_MAX_BINDINGS];
    VkSpecializationMapEntry spec_map[CRAW_MAX_SPEC_ENTRIES + 3];
    uint8_t spec_data[CRAW_MAX_SPEC_DATA + 3 * sizeof(uint32_t)];
    size_t spec_off;
    uint32_t nb_spec;
    bool bad;

    memset(sh, 0, sizeof(*sh));
    snprintf(sh->name, sizeof(sh->name), "%s", desc->name ? desc->name : "shader");

    /* Half a specialization block is a caller bug rather than a shader that
     * has none, and ids 253 and up belong to the workgroup size. */
    bad = !desc->spirv || !desc->spirv_len ||
          !desc->wg[0] || !desc->wg[1] || !desc->wg[2] ||
          desc->nb_bindings > CRAW_MAX_BINDINGS ||
          desc->nb_spec_entries > CRAW_MAX_SPEC_ENTRIES ||
          desc->spec_size > CRAW_MAX_SPEC_DATA ||
          desc->push_size > CRAW_MAX_PUSH_SIZE ||
          (!!desc->nb_spec_entries != !!desc->spec_size) ||
          (desc->nb_spec_entries && (!desc->spec_entries || !desc->spec_data));
    for (uint32_t i = 0; i < desc->nb_spec_entries && !bad; i++)
        bad = desc->spec_entries[i].constantID >= CRAW_SPEC_ID_WG_X;

    if (bad) {
        craw_log(ctx, CRAW_LOG_ERROR,
                 "%s: bad shader description (%zu bytes of SPIR-V, workgroup "
                 "%ux%ux%u, %u bindings, %u push bytes, %u spec constants)\n",
                 sh->name, desc->spirv_len, desc->wg[0], desc->wg[1], desc->wg[2],
                 desc->nb_bindings, desc->push_size, desc->nb_spec_entries);
        return CRAW_ERROR(EINVAL);
    }

    sh->wg[0] = desc->wg[0];
    sh->wg[1] = desc->wg[1];
    sh->wg[2] = desc->wg[2];
    sh->push_size = desc->push_size;

    for (uint32_t i = 0; i < desc->nb_bindings; i++) {
        bindings[i] = (VkDescriptorSetLayoutBinding) {
            .binding         = desc->bindings[i].binding,
            .descriptorType  = desc->bindings[i].type,
            .descriptorCount = 1,
            .stageFlags      = VK_SHADER_STAGE_COMPUTE_BIT,
        };
    }

    VkDescriptorSetLayoutCreateInfo set_info = {
        .sType        = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
        .flags        = VK_DESCRIPTOR_SET_LAYOUT_CREATE_PUSH_DESCRIPTOR_BIT,
        .bindingCount = desc->nb_bindings,
        .pBindings    = desc->nb_bindings ? bindings : NULL,
    };
    res = fn->CreateDescriptorSetLayout(ctx->device, &set_info, ctx->alloc,
                                        &sh->dset_layout);
    if (res != VK_SUCCESS) {
        craw_log(ctx, CRAW_LOG_ERROR,
                 "%s: unable to create a descriptor set layout: %s\n",
                 sh->name, craw_vk_ret2str(res));
        err = craw_vk_ret2err(res);
        goto fail;
    }

    VkPushConstantRange pc_range = {
        .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT,
        .offset     = 0,
        .size       = desc->push_size,
    };
    VkPipelineLayoutCreateInfo layout_info = {
        .sType                  = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO,
        .setLayoutCount         = 1,
        .pSetLayouts            = &sh->dset_layout,
        .pushConstantRangeCount = desc->push_size ? 1 : 0,
        .pPushConstantRanges    = desc->push_size ? &pc_range : NULL,
    };
    res = fn->CreatePipelineLayout(ctx->device, &layout_info, ctx->alloc,
                                   &sh->layout);
    if (res != VK_SUCCESS) {
        craw_log(ctx, CRAW_LOG_ERROR,
                 "%s: unable to create a pipeline layout: %s\n",
                 sh->name, craw_vk_ret2str(res));
        err = craw_vk_ret2err(res);
        goto fail;
    }

    VkShaderModuleCreateInfo mod_info = {
        .sType    = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO,
        .codeSize = desc->spirv_len,
        .pCode    = (const uint32_t *)desc->spirv,
    };
    res = fn->CreateShaderModule(ctx->device, &mod_info, ctx->alloc, &sh->module);
    if (res != VK_SUCCESS) {
        craw_log(ctx, CRAW_LOG_ERROR,
                 "%s: unable to create the shader module: %s\n",
                 sh->name, craw_vk_ret2str(res));
        err = craw_vk_ret2err(res);
        goto fail;
    }

    nb_spec = 0;
    if (desc->nb_spec_entries) {
        memcpy(spec_data, desc->spec_data, desc->spec_size);
        memcpy(spec_map, desc->spec_entries,
               desc->nb_spec_entries * sizeof(*spec_map));
        nb_spec = desc->nb_spec_entries;
        spec_off = craw_align4(desc->spec_size);
    } else {
        spec_off = 0;
    }

    memcpy(spec_data + spec_off, sh->wg, sizeof(sh->wg));
    for (uint32_t i = 0; i < 3; i++) {
        spec_map[nb_spec + i] = (VkSpecializationMapEntry) {
            .constantID = CRAW_SPEC_ID_WG_X + i,
            .offset     = (uint32_t)(spec_off + i * sizeof(uint32_t)),
            .size       = sizeof(uint32_t),
        };
    }
    nb_spec += 3;

    VkSpecializationInfo spec_info = {
        .mapEntryCount = nb_spec,
        .pMapEntries   = spec_map,
        .dataSize      = spec_off + sizeof(sh->wg),
        .pData         = spec_data,
    };

    VkPipelineShaderStageRequiredSubgroupSizeCreateInfo sg_info = {
        .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_REQUIRED_SUBGROUP_SIZE_CREATE_INFO,
        .requiredSubgroupSize = desc->required_subgroup_size,
    };

    VkComputePipelineCreateInfo pipe_info = {
        .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
        .stage = {
            .sType  = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
            .pNext  = desc->required_subgroup_size ? &sg_info : NULL,
            .flags  = desc->required_subgroup_size ?
                      VK_PIPELINE_SHADER_STAGE_CREATE_REQUIRE_FULL_SUBGROUPS_BIT : 0,
            .stage  = VK_SHADER_STAGE_COMPUTE_BIT,
            .module = sh->module,
            .pName  = "main",
            .pSpecializationInfo = &spec_info,
        },
        .layout = sh->layout,
    };
    res = fn->CreateComputePipelines(ctx->device, VK_NULL_HANDLE, 1, &pipe_info,
                                     ctx->alloc, &sh->pipeline);
    if (res != VK_SUCCESS) {
        craw_log(ctx, CRAW_LOG_ERROR,
                 "%s: unable to create the compute pipeline: %s\n",
                 sh->name, craw_vk_ret2str(res));
        err = craw_vk_ret2err(res);
        goto fail;
    }

    craw_log(ctx, CRAW_LOG_DEBUG,
             "Shader %s created, workgroup %ux%ux%u, %u descriptors, "
             "%u bytes of push constants\n",
             sh->name, sh->wg[0], sh->wg[1], sh->wg[2], desc->nb_bindings,
             sh->push_size);

    return 0;

fail:
    craw_vk_shader_free(ctx, sh);
    return err;
}

void craw_vk_shader_free(CRAWContext *ctx, CRAWVkShader *sh)
{
    const CRAWVkFunctions *fn = &ctx->fn;

    if (sh->pipeline)
        fn->DestroyPipeline(ctx->device, sh->pipeline, ctx->alloc);
    if (sh->module)
        fn->DestroyShaderModule(ctx->device, sh->module, ctx->alloc);
    if (sh->layout)
        fn->DestroyPipelineLayout(ctx->device, sh->layout, ctx->alloc);
    if (sh->dset_layout)
        fn->DestroyDescriptorSetLayout(ctx->device, sh->dset_layout, ctx->alloc);

    memset(sh, 0, sizeof(*sh));
}

void craw_vk_desc_image(CRAWVkDescWrite *w, uint32_t binding, const CRAWVkImage *img)
{
    *w = (CRAWVkDescWrite) {
        .img = {
            .imageView   = img->view,
            .imageLayout = img->layout,
        },
        .write = {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding      = binding,
            .descriptorCount = 1,
            .descriptorType  = VK_DESCRIPTOR_TYPE_STORAGE_IMAGE,
        },
    };
}

void craw_vk_desc_buffer(CRAWVkDescWrite *w, uint32_t binding, VkDescriptorType type,
                         const CRAWVkBuffer *buf, VkDeviceSize offset,
                         VkDeviceSize range)
{
    *w = (CRAWVkDescWrite) {
        .buf = {
            .buffer = buf->buf,
            .offset = offset,
            .range  = range ? range : VK_WHOLE_SIZE,
        },
        .write = {
            .sType           = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
            .dstBinding      = binding,
            .descriptorCount = 1,
            .descriptorType  = type,
        },
    };
}

void craw_vk_shader_bind(CRAWContext *ctx, VkCommandBuffer cmd, const CRAWVkShader *sh)
{
    ctx->fn.CmdBindPipeline(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sh->pipeline);
}

void craw_vk_shader_push_desc(CRAWContext *ctx, VkCommandBuffer cmd,
                              const CRAWVkShader *sh, CRAWVkDescWrite *writes,
                              uint32_t nb_writes)
{
    VkWriteDescriptorSet flat[CRAW_MAX_BINDINGS];

    if (nb_writes > CRAW_MAX_BINDINGS) {
        craw_log(ctx, CRAW_LOG_ERROR, "%s: %u descriptors, at most %u fit\n",
                 sh->name, nb_writes, CRAW_MAX_BINDINGS);
        nb_writes = CRAW_MAX_BINDINGS;
    }

    /* The info structs live inside each CRAWVkDescWrite so that callers can
     * build the array on their stack without tracking two of them. */
    for (uint32_t i = 0; i < nb_writes; i++) {
        switch (writes[i].write.descriptorType) {
        case VK_DESCRIPTOR_TYPE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER:
        case VK_DESCRIPTOR_TYPE_SAMPLED_IMAGE:
        case VK_DESCRIPTOR_TYPE_STORAGE_IMAGE:
        case VK_DESCRIPTOR_TYPE_INPUT_ATTACHMENT:
            writes[i].write.pImageInfo = &writes[i].img;
            break;
        default:
            writes[i].write.pBufferInfo = &writes[i].buf;
            break;
        }
        flat[i] = writes[i].write;
    }

    ctx->fn.CmdPushDescriptorSet(cmd, VK_PIPELINE_BIND_POINT_COMPUTE, sh->layout,
                                 0, nb_writes, flat);
}

void craw_vk_shader_push_const(CRAWContext *ctx, VkCommandBuffer cmd,
                               const CRAWVkShader *sh, const void *data,
                               uint32_t size)
{
    if (!size || !data || !sh->push_size)
        return;

    ctx->fn.CmdPushConstants(cmd, sh->layout, VK_SHADER_STAGE_COMPUTE_BIT, 0,
                             CRAW_MIN(size, sh->push_size), data);
}

void craw_vk_shader_dispatch(CRAWContext *ctx, VkCommandBuffer cmd,
                             const CRAWVkShader *sh, uint32_t w, uint32_t h,
                             uint32_t d)
{
    ctx->fn.CmdDispatch(cmd,
                        (w + sh->wg[0] - 1) / sh->wg[0],
                        (h + sh->wg[1] - 1) / sh->wg[1],
                        (d + sh->wg[2] - 1) / sh->wg[2]);
}
