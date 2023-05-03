#include <set>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <NvInferPlugin.h>

#include "plugin/gather_csr_plugin.h"
#include "gather_csr.h"


namespace tensorrt_scatter
{
namespace plugin
{

namespace
{
const char* const PLUGIN_VERSION{"1"};
const char* const PLUGIN_NAME{"TRTS_GatherCSR"};
}  // namespace

// IPluginV2 Methods
const nvinfer1::AsciiChar* GatherCSRPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const nvinfer1::AsciiChar* GatherCSRPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int32_t GatherCSRPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t GatherCSRPlugin::initialize() noexcept
{
    return 0;
}

void GatherCSRPlugin::terminate() noexcept
{
    return;
}

size_t GatherCSRPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void GatherCSRPlugin::serialize(void* buffer) const noexcept
{
    return;
}

void GatherCSRPlugin::destroy() noexcept
{
    delete this;
}

void GatherCSRPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const nvinfer1::AsciiChar* GatherCSRPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// IPluginV2Ext Methods
nvinfer1::DataType GatherCSRPlugin::getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, 
    int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* GatherCSRPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new GatherCSRPlugin();
        plugin->setPluginNamespace(plugin->getPluginNamespace());
        return plugin;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return nullptr;
}

nvinfer1::DimsExprs GatherCSRPlugin::getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{    
    return inputs[2];
}

bool GatherCSRPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // src, indptr, base, out
    const nvinfer1::PluginTensorDesc desc = inOut[pos];

    if (desc.format != nvinfer1::TensorFormat::kLINEAR)
        return false;

    if (pos == 0)  // src
        return desc.type == nvinfer1::DataType::kFLOAT || desc.type == nvinfer1::DataType::kHALF;
    if (pos == 1)  // indptr
        return desc.type == nvinfer1::DataType::kINT32;
    if (pos == 2)  // base
        return desc.type == inOut[0].type;
    if (pos == 3)  // out
        return desc.type == inOut[0].type;
    return false;
}

void GatherCSRPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs, 
    const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept {}

size_t GatherCSRPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, 
    const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept
{
    return 0;  // TODO: check if this is correct
}

int32_t GatherCSRPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, 
    void* workspace, cudaStream_t stream) noexcept
{
    int32_t status = -1;

    try
    {   
        // src, indptr, base, out
        auto indptr = static_cast<const int32_t*>(inputs[1]);
        std::vector<int32_t> src_size(inputDesc[0].dims.d, inputDesc[0].dims.d + inputDesc[0].dims.nbDims);
        std::vector<int32_t> indptr_size(inputDesc[1].dims.d, inputDesc[1].dims.d + inputDesc[1].dims.nbDims);
        std::vector<int32_t> out_size(outputDesc[0].dims.d, outputDesc[0].dims.d + outputDesc[0].dims.nbDims);
        
        if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
        {
            auto src = static_cast<const float*>(inputs[0]);
            auto base = static_cast<const float*>(inputs[2]);
            auto out = static_cast<float*>(outputs[0]);
            status = gather_csr_launch<float>(src, src_size, indptr, indptr_size, base, out, out_size, stream);
        }
        else if (inputDesc[0].type == nvinfer1::DataType::kHALF)
        {
            auto src = static_cast<const half*>(inputs[0]);
            auto base = static_cast<const half*>(inputs[2]);
            auto out = static_cast<half*>(outputs[0]);
            status = gather_csr_launch<half>(src, src_size, indptr, indptr_size, base, out, out_size, stream);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return status;
}

// GatherCSRPlugin Methods
GatherCSRPlugin::GatherCSRPlugin() {}

GatherCSRPlugin::GatherCSRPlugin(const void* data, size_t length) {}

REGISTER_TENSORRT_PLUGIN(GatherCSRPluginCreator);

// Static class fields initialization
nvinfer1::PluginFieldCollection GatherCSRPluginCreator::mFC{};

// IPluginCreator Methods
const nvinfer1::AsciiChar* GatherCSRPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const nvinfer1::AsciiChar* GatherCSRPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* GatherCSRPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV2* GatherCSRPluginCreator::createPlugin(const nvinfer1::AsciiChar* name, const nvinfer1::PluginFieldCollection* fc) noexcept
{
    try
    {
        nvinfer1::IPluginV2* plugin = new GatherCSRPlugin();
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << '\n';
    }
    return nullptr;
}

nvinfer1::IPluginV2* GatherCSRPluginCreator::deserializePlugin(const nvinfer1::AsciiChar* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed,
        nvinfer1::IPluginV2* plugin = new GatherCSRPlugin(serialData, serialLength);
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return nullptr;
    
}

void GatherCSRPluginCreator::setPluginNamespace(const nvinfer1::AsciiChar* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const nvinfer1::AsciiChar* GatherCSRPluginCreator::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// GatherCSRPluginCreator Methods
GatherCSRPluginCreator::GatherCSRPluginCreator()
{
    mFC.nbFields = 0;
}

}  // namespace plugin
}  // namespace tensorrt_scatter
