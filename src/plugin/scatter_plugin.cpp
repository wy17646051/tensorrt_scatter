#include <iostream>
#include <set>
#include <vector>
#include <tuple>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <NvInferPlugin.h>

#include "common/reduction.h"
#include "plugin/scatter_plugin.h"
#include "scatter.h"

#include <iostream>

namespace tensorrt_scatter
{
namespace plugin
{

namespace
{
const char* const PLUGIN_VERSION{"1"};
const char* const PLUGIN_NAME{"TRTS_Scatter"};
}  // namespace

// IPluginV2 Methods
const nvinfer1::AsciiChar* ScatterPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const nvinfer1::AsciiChar* ScatterPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int32_t ScatterPlugin::getNbOutputs() const noexcept
{
    if (mReduce == ReductionType::MAX || mReduce == ReductionType::MIN)
        return 2;
    else
        return 1;
}

int32_t ScatterPlugin::initialize() noexcept
{
    return 0;
}

void ScatterPlugin::terminate() noexcept
{
    return;
}

size_t ScatterPlugin::getSerializationSize() const noexcept
{
    return sizeof(int32_t) + sizeof(size_t) + sizeof(ReductionType);
}

void ScatterPlugin::serialize(void* buffer) const noexcept
{
    char* _buffer = reinterpret_cast<char*>(buffer);
    *reinterpret_cast<int32_t*>(_buffer) = mDim;
    _buffer += sizeof(int32_t);
    *reinterpret_cast<size_t*>(_buffer) = mDimSize;
    _buffer += sizeof(size_t);
    *reinterpret_cast<ReductionType*>(_buffer) = mReduce;
}

void ScatterPlugin::destroy() noexcept
{
    delete this;
}

void ScatterPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const nvinfer1::AsciiChar* ScatterPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// IPluginV2Ext Methods
nvinfer1::DataType ScatterPlugin::getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, 
    int32_t nbInputs) const noexcept
{
    if (index == 0)
        return inputTypes[0];
    else
        return nvinfer1::DataType::kINT32;
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* ScatterPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new ScatterPlugin(mDim, mDimSize, mReduce);
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

nvinfer1::DimsExprs ScatterPlugin::getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs output;
    
    output.nbDims = inputs[0].nbDims;

    auto index_dim = mDim < 0 ? inputs[0].nbDims + mDim : mDim;
    for (auto i = 0; i < inputs[0].nbDims; i++)
        if (i == index_dim)
            output.d[i] = exprBuilder.constant(mDimSize);
        else
            output.d[i] = inputs[0].d[i];
    
    return output;
}

bool ScatterPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // src, index, (base), out (arg_out)
    const nvinfer1::PluginTensorDesc desc = inOut[pos];

    if (desc.format != nvinfer1::TensorFormat::kLINEAR)
        return false;

    if (pos == 0)  // src
        return desc.type == nvinfer1::DataType::kFLOAT || desc.type == nvinfer1::DataType::kHALF;
    if (pos == 1)  // index
        return desc.type == nvinfer1::DataType::kINT32;
    if (pos == 2)  // base | out
        return desc.type == inOut[0].type;
    if (pos == 3)  // out | arg_out
        if (nbInputs == 3)  // out
            return desc.type == inOut[0].type;
        else // arg_out
            return desc.type == nvinfer1::DataType::kINT32;
    if (pos == 4)  // arg_out
        return desc.type == nvinfer1::DataType::kINT32;
    return false;
}

void ScatterPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs, 
    const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    if (nbInputs == 3)
        setWithBase(true);
    else
        setWithBase(false);
    return;
}

size_t ScatterPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, 
    const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept
{
    return 0;  // TODO: check if this is correct
}

int32_t ScatterPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, 
    void* workspace, cudaStream_t stream) noexcept
{
    int32_t status = -1;
    
    try
    {   
        // src, index, (base), out (arg_out)
        auto index = static_cast<const int32_t*>(inputs[1]);
        int32_t* arg_out = getNbOutputs() == 2 ? static_cast<int32_t*>(outputs[1]) : nullptr;
        
        std::vector<int32_t> src_size(inputDesc[0].dims.d, inputDesc[0].dims.d + inputDesc[0].dims.nbDims);
        std::vector<int32_t> out_size(outputDesc[0].dims.d, outputDesc[0].dims.d + outputDesc[0].dims.nbDims);
        
        if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
        {
            auto src = static_cast<const float*>(inputs[0]);
            std::tuple<float*, int32_t*> out = std::make_tuple(static_cast<float*>(outputs[0]), arg_out);

            AT_DISPATCH_REDUCTION_TYPES(REDUCE2reduce.at(mReduce), [&] {
                if (getWithBase())
                {
                    const float* base = static_cast<const float*>(inputs[2]);
                    status = scatter_launch<float, REDUCE>(src, src_size, index, mDim, base, out, out_size, stream);
                }
                else
                    status = scatter_launch<float, REDUCE>(src, src_size, index, mDim, out, out_size, stream);
            });
        }
        else if (inputDesc[0].type == nvinfer1::DataType::kHALF)
        {
            auto src = static_cast<const half*>(inputs[0]);            
            std::tuple<half*, int32_t*> out = std::make_tuple(static_cast<half*>(outputs[0]), arg_out);

            AT_DISPATCH_REDUCTION_TYPES(REDUCE2reduce.at(mReduce), [&] {
                if (getWithBase())
                {
                    const half* base = static_cast<const half*>(inputs[2]);
                    status = scatter_launch<half, REDUCE>(src, src_size, index, mDim, base, out, out_size, stream);
                }
                else
                    status = scatter_launch<half, REDUCE>(src, src_size, index, mDim, out, out_size, stream);
            });
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return status;
}

// ScatterPlugin Methods
ScatterPlugin::ScatterPlugin(int32_t dim, size_t dimSize, ReductionType reduce):
    mDim(dim), mDimSize(dimSize), mReduce(reduce) {}

ScatterPlugin::ScatterPlugin(const void* data, size_t length)
{
    const char* _data = reinterpret_cast<const char*>(data);
    mDim = *reinterpret_cast<const int32_t*>(_data);
    _data += sizeof(int32_t);
    mDimSize = *reinterpret_cast<const size_t*>(_data);
    _data += sizeof(size_t);
    mReduce = *reinterpret_cast<const ReductionType*>(_data);
}

void ScatterPlugin::setWithBase(bool withBase) noexcept
{
    mWithBase = withBase;
}

bool ScatterPlugin::getWithBase() const noexcept
{
    return mWithBase;
}

REGISTER_TENSORRT_PLUGIN(ScatterPluginCreator);

// Static class fields initialization
nvinfer1::PluginFieldCollection ScatterPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> ScatterPluginCreator::mPluginAttributes;

// IPluginCreator Methods
const nvinfer1::AsciiChar* ScatterPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const nvinfer1::AsciiChar* ScatterPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* ScatterPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV2* ScatterPluginCreator::createPlugin(const nvinfer1::AsciiChar* name, const nvinfer1::PluginFieldCollection* fc) noexcept
{
    try
    {
        const nvinfer1::PluginField* fields = fc->fields;

        int32_t dim;
        size_t dimSize;
        ReductionType reduce;
        for (auto i = 0; i < fc->nbFields; i++)
        {
            const nvinfer1::PluginField& field = fields[i];
            if (!strcmp(field.name, "dim"))
            {
                dim = *(static_cast<const int32_t*>(field.data));
            }
            else if (!strcmp(field.name, "dim_size"))
            {
                dimSize = *(static_cast<const size_t*>(field.data));
            }
            else if (!strcmp(field.name, "reduce"))
            {
                reduce = reduce2REDUCE.at(static_cast<const char*>(field.data));
            }
        }

        nvinfer1::IPluginV2* plugin = new ScatterPlugin(dim, dimSize, reduce);
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << '\n';
    }
    return nullptr;
}

nvinfer1::IPluginV2* ScatterPluginCreator::deserializePlugin(const nvinfer1::AsciiChar* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed,
        nvinfer1::IPluginV2* plugin = new ScatterPlugin(serialData, serialLength);
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return nullptr;
    
}

void ScatterPluginCreator::setPluginNamespace(const nvinfer1::AsciiChar* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const nvinfer1::AsciiChar* ScatterPluginCreator::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// ScatterPluginCreator Methods
ScatterPluginCreator::ScatterPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("dim", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("dim_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("reduce", nullptr, nvinfer1::PluginFieldType::kCHAR, 1));
 
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

}  // namespace plugin
}  // namespace tensorrt_scatter
