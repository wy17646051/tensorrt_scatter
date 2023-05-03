#include <iostream>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <NvInferPlugin.h>

#include "common/reduction.h"
#include "plugin/segment_coo_plugin.h"
#include "segment_coo.h"


namespace tensorrt_scatter
{
namespace plugin
{

namespace
{
const char* const PLUGIN_VERSION{"1"};
const char* const PLUGIN_NAME{"TRTS_SegmentCOO"};
}  // namespace

// IPluginV2 Methods
const nvinfer1::AsciiChar* SegmentCOOPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const nvinfer1::AsciiChar* SegmentCOOPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int32_t SegmentCOOPlugin::getNbOutputs() const noexcept
{
    if (mReduce == ReductionType::MAX || mReduce == ReductionType::MIN)
        return 2;
    else
        return 1;
}

int32_t SegmentCOOPlugin::initialize() noexcept
{
    return 0;
}

void SegmentCOOPlugin::terminate() noexcept
{
    return;
}

size_t SegmentCOOPlugin::getSerializationSize() const noexcept
{
    return sizeof(size_t) + sizeof(ReductionType);
}

void SegmentCOOPlugin::serialize(void* buffer) const noexcept
{
    char* _buffer = reinterpret_cast<char*>(buffer);
    *reinterpret_cast<size_t*>(_buffer) = mDimSize;
    _buffer += sizeof(size_t);
    *reinterpret_cast<ReductionType*>(_buffer) = mReduce;
}

void SegmentCOOPlugin::destroy() noexcept
{
    delete this;
}

void SegmentCOOPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const nvinfer1::AsciiChar* SegmentCOOPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// IPluginV2Ext Methods
nvinfer1::DataType SegmentCOOPlugin::getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, 
    int32_t nbInputs) const noexcept
{
    if (index == 0)
        return inputTypes[0];
    else
        return nvinfer1::DataType::kINT32;
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* SegmentCOOPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new SegmentCOOPlugin(mDimSize, mReduce);
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

nvinfer1::DimsExprs SegmentCOOPlugin::getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs output;
    
    output.nbDims = inputs[0].nbDims;

    auto index_dim = inputs[1].nbDims - 1;
    for (auto i = 0; i < inputs[0].nbDims; i++)
        if (i == index_dim)
            output.d[i] = exprBuilder.constant(mDimSize);
        else
            output.d[i] = inputs[0].d[i];
    
    return output;
}

bool SegmentCOOPlugin::supportsFormatCombination(
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

void SegmentCOOPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs, 
    const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    if (nbInputs == 3)
        setWithBase(true);
    else
        setWithBase(false);
    return;
}

size_t SegmentCOOPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, 
    const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept
{
    return 0;  // TODO: check if this is correct
}

int32_t SegmentCOOPlugin::enqueue(
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
        std::vector<int32_t> index_size(inputDesc[1].dims.d, inputDesc[1].dims.d + inputDesc[1].dims.nbDims);
        std::vector<int32_t> out_size(outputDesc[0].dims.d, outputDesc[0].dims.d + outputDesc[0].dims.nbDims);
        
        if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
        {
            auto src = static_cast<const float*>(inputs[0]);
            std::tuple<float*, int32_t*> out = std::make_tuple(static_cast<float*>(outputs[0]), arg_out);

            AT_DISPATCH_REDUCTION_TYPES(REDUCE2reduce.at(mReduce), [&] {
                if (getWithBase())
                {
                    const float* base = static_cast<const float*>(inputs[2]);
                    status = segment_coo_launch<float, REDUCE>(src, src_size, index, index_size, base, out, out_size, stream);
                }
                else
                    status = segment_coo_launch<float, REDUCE>(src, src_size, index, index_size, out, out_size, stream);
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
                    status = segment_coo_launch<half, REDUCE>(src, src_size, index, index_size, base, out, out_size, stream);
                }
                else
                    status = segment_coo_launch<half, REDUCE>(src, src_size, index, index_size, out, out_size, stream);
            });
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return status;
}

// SegmentCOOPlugin Methods
SegmentCOOPlugin::SegmentCOOPlugin(size_t dimSize, ReductionType reduce): mDimSize(dimSize), mReduce(reduce) {}

SegmentCOOPlugin::SegmentCOOPlugin(const void* data, size_t length)
{
    const char* _data = reinterpret_cast<const char*>(data);
    mDimSize = *reinterpret_cast<const size_t*>(_data);
    _data += sizeof(size_t);
    mReduce = *reinterpret_cast<const ReductionType*>(_data);
}

void SegmentCOOPlugin::setWithBase(bool withBase) noexcept
{
    mWithBase = withBase;
}

bool SegmentCOOPlugin::getWithBase() const noexcept
{
    return mWithBase;
}

REGISTER_TENSORRT_PLUGIN(SegmentCOOPluginCreator);

// Static class fields initialization
nvinfer1::PluginFieldCollection SegmentCOOPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> SegmentCOOPluginCreator::mPluginAttributes;

// IPluginCreator Methods
const nvinfer1::AsciiChar* SegmentCOOPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const nvinfer1::AsciiChar* SegmentCOOPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* SegmentCOOPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV2* SegmentCOOPluginCreator::createPlugin(const nvinfer1::AsciiChar* name, const nvinfer1::PluginFieldCollection* fc) noexcept
{
    try
    {
        const nvinfer1::PluginField* fields = fc->fields;

        size_t dimSize;
        ReductionType reduce;
        for (auto i = 0; i < fc->nbFields; i++)
        {
            const nvinfer1::PluginField& field = fields[i];
            if (!strcmp(field.name, "dim_size"))
            {
                dimSize = *(static_cast<const size_t*>(field.data));
            }
            else if (!strcmp(field.name, "reduce"))
            {
                reduce = reduce2REDUCE.at(static_cast<const char*>(field.data));
            }
        }

        nvinfer1::IPluginV2* plugin = new SegmentCOOPlugin(dimSize, reduce);
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << '\n';
    }
    return nullptr;
}

nvinfer1::IPluginV2* SegmentCOOPluginCreator::deserializePlugin(const nvinfer1::AsciiChar* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed,
        nvinfer1::IPluginV2* plugin = new SegmentCOOPlugin(serialData, serialLength);
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return nullptr;
    
}

void SegmentCOOPluginCreator::setPluginNamespace(const nvinfer1::AsciiChar* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const nvinfer1::AsciiChar* SegmentCOOPluginCreator::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// SegmentCOOPluginCreator Methods
SegmentCOOPluginCreator::SegmentCOOPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("dim_size", nullptr, nvinfer1::PluginFieldType::kINT32, 1));
    mPluginAttributes.emplace_back(nvinfer1::PluginField("reduce", nullptr, nvinfer1::PluginFieldType::kCHAR, 1));
 
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

}  // namespace plugin
}  // namespace tensorrt_scatter
