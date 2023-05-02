#include <set>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <NvInferPlugin.h>

#include "plugin/gather_coo_plugin.h"
#include "gather_coo.h"


namespace tensorrt_scatter
{
namespace plugin
{

namespace
{
const char* const PLUGIN_VERSION{"1"};
const char* const PLUGIN_NAME{"TRTS_GatherCOO"};
}  // namespace

// IPluginV2 Methods
const nvinfer1::AsciiChar* GatherCOOPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const nvinfer1::AsciiChar* GatherCOOPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int32_t GatherCOOPlugin::getNbOutputs() const noexcept
{
    return 1;
}

int32_t GatherCOOPlugin::initialize() noexcept
{
    return 0;
}

void GatherCOOPlugin::terminate() noexcept
{
    return;
}

size_t GatherCOOPlugin::getSerializationSize() const noexcept
{
    return 0;
}

void GatherCOOPlugin::serialize(void* buffer) const noexcept
{
    return;
}

void GatherCOOPlugin::destroy() noexcept
{
    delete this;
}

void GatherCOOPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const nvinfer1::AsciiChar* GatherCOOPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// IPluginV2Ext Methods
nvinfer1::DataType GatherCOOPlugin::getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, 
    int32_t nbInputs) const noexcept
{
    return inputTypes[0];
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* GatherCOOPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new GatherCOOPlugin();
        plugin->setPluginNamespace(plugin->getPluginNamespace());
        plugin->setWithBase(plugin->getWithBase());
        return plugin;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return nullptr;
}

nvinfer1::DimsExprs GatherCOOPlugin::getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs output;
    
    output.nbDims = inputs[0].nbDims;

    auto index_dim = inputs[1].nbDims - 1;
    for (auto i = 0; i < inputs[0].nbDims; i++)
        if (i == index_dim)
            output.d[i] = inputs[1].d[i];
        else
            output.d[i] = inputs[0].d[i];
    
    return output;
}

bool GatherCOOPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // src, index, (base), out
    const nvinfer1::PluginTensorDesc desc = inOut[pos];

    if (desc.format != nvinfer1::TensorFormat::kLINEAR)
        return false;

    if (pos == 0)  // src
        return desc.type == nvinfer1::DataType::kFLOAT || desc.type == nvinfer1::DataType::kHALF;
    if (pos == 1)  // index
        return desc.type == nvinfer1::DataType::kINT32;
    if (pos == 2)  // base | out
        return desc.type == inOut[0].type;
    if (pos == 3)  // out
        return desc.type == inOut[0].type;
    return false;
}

void GatherCOOPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs, 
    const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    if (nbInputs == 3)
        setWithBase(true);
    else
        setWithBase(false);
    return;
}

size_t GatherCOOPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, 
    const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept
{
    return 0;  // TODO: check if this is correct
}

int32_t GatherCOOPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, 
    void* workspace, cudaStream_t stream) noexcept
{
    int32_t status = -1;

    try
    {   
        // src, index, (base), out
        auto index = static_cast<const int32_t*>(inputs[1]);
        std::vector<int32_t> src_size(inputDesc[0].dims.d, inputDesc[0].dims.d + inputDesc[0].dims.nbDims);
        std::vector<int32_t> index_size(inputDesc[1].dims.d, inputDesc[1].dims.d + inputDesc[1].dims.nbDims);
        
        if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
        {
            auto src = static_cast<const float*>(inputs[0]);
            auto out = static_cast<float*>(outputs[0]);

            if (getWithBase())
            {
                const float* base = static_cast<const float*>(inputs[2]);
                status = gather_coo_launch<float>(src, src_size, index, index_size, base, out, stream);
            }
            else
                status = gather_coo_launch<float>(src, src_size, index, index_size, out, stream);
        }
        else if (inputDesc[0].type == nvinfer1::DataType::kHALF)
        {
            auto src = static_cast<const half*>(inputs[0]);            
            auto out = static_cast<half*>(outputs[0]);

            if (getWithBase())
            {
                const half* base = static_cast<const half*>(inputs[2]);
                status = gather_coo_launch<half>(src, src_size, index, index_size, base, out, stream);
            }
            else
                status = gather_coo_launch<half>(src, src_size, index, index_size, out, stream);
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return status;
}

// GatherCOOPlugin Methods
GatherCOOPlugin::GatherCOOPlugin() {}

GatherCOOPlugin::GatherCOOPlugin(const void* data, size_t length) {}

void GatherCOOPlugin::setWithBase(bool withBase) noexcept
{
    mWithBase = withBase;
}

bool GatherCOOPlugin::getWithBase() const noexcept
{
    return mWithBase;
}

REGISTER_TENSORRT_PLUGIN(GatherCOOPluginCreator);

// Static class fields initialization
nvinfer1::PluginFieldCollection GatherCOOPluginCreator::mFC{};

// IPluginCreator Methods
const nvinfer1::AsciiChar* GatherCOOPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const nvinfer1::AsciiChar* GatherCOOPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* GatherCOOPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV2* GatherCOOPluginCreator::createPlugin(const nvinfer1::AsciiChar* name, const nvinfer1::PluginFieldCollection* fc) noexcept
{
    try
    {
        nvinfer1::IPluginV2* plugin = new GatherCOOPlugin();
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << '\n';
    }
    return nullptr;
}

nvinfer1::IPluginV2* GatherCOOPluginCreator::deserializePlugin(const nvinfer1::AsciiChar* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed,
        nvinfer1::IPluginV2* plugin = new GatherCOOPlugin(serialData, serialLength);
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return nullptr;
    
}

void GatherCOOPluginCreator::setPluginNamespace(const nvinfer1::AsciiChar* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const nvinfer1::AsciiChar* GatherCOOPluginCreator::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// GatherCOOPluginCreator Methods
GatherCOOPluginCreator::GatherCOOPluginCreator()
{
    mFC.nbFields = 0;
}

}  // namespace plugin
}  // namespace tensorrt_scatter
