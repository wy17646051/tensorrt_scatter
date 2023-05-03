#include <iostream>
#include <tuple>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <NvInferPlugin.h>

#include "common/reduction.h"
#include "plugin/segment_csr_plugin.h"
#include "segment_csr.h"


namespace tensorrt_scatter
{
namespace plugin
{

namespace
{
const char* const PLUGIN_VERSION{"1"};
const char* const PLUGIN_NAME{"TRTS_SegmentCSR"};
}  // namespace

// IPluginV2 Methods
const nvinfer1::AsciiChar* SegmentCSRPlugin::getPluginType() const noexcept
{
    return PLUGIN_NAME;
}

const nvinfer1::AsciiChar* SegmentCSRPlugin::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

int32_t SegmentCSRPlugin::getNbOutputs() const noexcept
{
    if (mReduce == ReductionType::MAX || mReduce == ReductionType::MIN)
        return 2;
    else
        return 1;
}

int32_t SegmentCSRPlugin::initialize() noexcept
{
    return 0;
}

void SegmentCSRPlugin::terminate() noexcept
{
    return;
}

size_t SegmentCSRPlugin::getSerializationSize() const noexcept
{
    return sizeof(ReductionType);
}

void SegmentCSRPlugin::serialize(void* buffer) const noexcept
{
    *reinterpret_cast<ReductionType*>(buffer) = mReduce;
}

void SegmentCSRPlugin::destroy() noexcept
{
    delete this;
}

void SegmentCSRPlugin::setPluginNamespace(const char* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const nvinfer1::AsciiChar* SegmentCSRPlugin::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// IPluginV2Ext Methods
nvinfer1::DataType SegmentCSRPlugin::getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, 
    int32_t nbInputs) const noexcept
{
    if (index == 0)
        return inputTypes[0];
    else
        return nvinfer1::DataType::kINT32;
}

// IPluginV2DynamicExt Methods
nvinfer1::IPluginV2DynamicExt* SegmentCSRPlugin::clone() const noexcept
{
    try
    {
        auto* plugin = new SegmentCSRPlugin(mReduce);
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

nvinfer1::DimsExprs SegmentCSRPlugin::getOutputDimensions(
    int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept
{
    nvinfer1::DimsExprs output;
    
    output.nbDims = inputs[0].nbDims;

    auto indptr_dim = inputs[1].nbDims - 1;
    for (auto i = 0; i < inputs[0].nbDims; i++)
        if (i == indptr_dim)
            output.d[i] = exprBuilder.constant(std::max<int32_t>(inputs[1].d[i]->getConstantValue(), 0));
        else
            output.d[i] = inputs[0].d[i];
    
    return output;
}

bool SegmentCSRPlugin::supportsFormatCombination(
    int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept
{
    // src, indptr, (base), out (arg_out)
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

void SegmentCSRPlugin::configurePlugin(const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs, 
    const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept
{
    if (nbInputs == 3)
        setWithBase(true);
    else
        setWithBase(false);
    return;
}

size_t SegmentCSRPlugin::getWorkspaceSize(const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, 
    const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept
{
    return 0;  // TODO: check if this is correct
}

int32_t SegmentCSRPlugin::enqueue(
    const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, 
    void* workspace, cudaStream_t stream) noexcept
{
    int32_t status = -1;
    
    try
    {   
        // src, indptr, (base), out (arg_out)
        auto indptr = static_cast<const int32_t*>(inputs[1]);
        int32_t* arg_out = getNbOutputs() == 2 ? static_cast<int32_t*>(outputs[1]) : nullptr;
        
        std::vector<int32_t> src_size(inputDesc[0].dims.d, inputDesc[0].dims.d + inputDesc[0].dims.nbDims);
        std::vector<int32_t> indptr_size(inputDesc[1].dims.d, inputDesc[1].dims.d + inputDesc[1].dims.nbDims);
        
        if (inputDesc[0].type == nvinfer1::DataType::kFLOAT)
        {
            auto src = static_cast<const float*>(inputs[0]);
            std::tuple<float*, int32_t*> out = std::make_tuple(static_cast<float*>(outputs[0]), arg_out);

            AT_DISPATCH_REDUCTION_TYPES(REDUCE2reduce.at(mReduce), [&] {
                if (getWithBase())
                {
                    const float* base = static_cast<const float*>(inputs[2]);
                    status = segment_csr_launch<float, REDUCE>(src, src_size, indptr, indptr_size, base, out, stream);
                }
                else
                    status = segment_csr_launch<float, REDUCE>(src, src_size, indptr, indptr_size, out, stream);
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
                    status = segment_csr_launch<half, REDUCE>(src, src_size, indptr, indptr_size, base, out, stream);
                }
                else
                    status = segment_csr_launch<half, REDUCE>(src, src_size, indptr, indptr_size, out, stream);
            });
        }
    }
    catch (const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    
    return status;
}

// SegmentCSRPlugin Methods
SegmentCSRPlugin::SegmentCSRPlugin(ReductionType reduce): mReduce(reduce) {}

SegmentCSRPlugin::SegmentCSRPlugin(const void* data, size_t length)
{
    mReduce = *reinterpret_cast<const ReductionType*>(data);
}

void SegmentCSRPlugin::setWithBase(bool withBase) noexcept
{
    mWithBase = withBase;
}

bool SegmentCSRPlugin::getWithBase() const noexcept
{
    return mWithBase;
}

REGISTER_TENSORRT_PLUGIN(SegmentCSRPluginCreator);

// Static class fields initialization
nvinfer1::PluginFieldCollection SegmentCSRPluginCreator::mFC{};
std::vector<nvinfer1::PluginField> SegmentCSRPluginCreator::mPluginAttributes;

// IPluginCreator Methods
const nvinfer1::AsciiChar* SegmentCSRPluginCreator::getPluginName() const noexcept
{
    return PLUGIN_NAME;
}

const nvinfer1::AsciiChar* SegmentCSRPluginCreator::getPluginVersion() const noexcept
{
    return PLUGIN_VERSION;
}

const nvinfer1::PluginFieldCollection* SegmentCSRPluginCreator::getFieldNames() noexcept
{
    return &mFC;
}

nvinfer1::IPluginV2* SegmentCSRPluginCreator::createPlugin(const nvinfer1::AsciiChar* name, const nvinfer1::PluginFieldCollection* fc) noexcept
{
    try
    {
        const nvinfer1::PluginField* fields = fc->fields;

        size_t dimSize;
        ReductionType reduce;
        for (auto i = 0; i < fc->nbFields; i++)
        {
            const nvinfer1::PluginField& field = fields[i];
            if (!strcmp(field.name, "reduce"))
                reduce = reduce2REDUCE.at(static_cast<const char*>(field.data));
        }

        nvinfer1::IPluginV2* plugin = new SegmentCSRPlugin(reduce);
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch (std::exception const& e)
    {
        std::cerr << e.what() << '\n';
    }
    return nullptr;
}

nvinfer1::IPluginV2* SegmentCSRPluginCreator::deserializePlugin(const nvinfer1::AsciiChar* name, const void* serialData, size_t serialLength) noexcept
{
    try
    {
        // This object will be deleted when the network is destroyed,
        nvinfer1::IPluginV2* plugin = new SegmentCSRPlugin(serialData, serialLength);
        plugin->setPluginNamespace(getPluginNamespace());
        return plugin;
    }
    catch(const std::exception& e)
    {
        std::cerr << e.what() << '\n';
    }
    return nullptr;
    
}

void SegmentCSRPluginCreator::setPluginNamespace(const nvinfer1::AsciiChar* pluginNamespace) noexcept
{
    mPluginNamespace = pluginNamespace;
}

const nvinfer1::AsciiChar* SegmentCSRPluginCreator::getPluginNamespace() const noexcept
{
    return mPluginNamespace.c_str();
}

// SegmentCSRPluginCreator Methods
SegmentCSRPluginCreator::SegmentCSRPluginCreator()
{
    mPluginAttributes.clear();
    mPluginAttributes.emplace_back(nvinfer1::PluginField("reduce", nullptr, nvinfer1::PluginFieldType::kCHAR, 1));
 
    mFC.nbFields = mPluginAttributes.size();
    mFC.fields = mPluginAttributes.data();
}

}  // namespace plugin
}  // namespace tensorrt_scatter
