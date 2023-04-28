#ifndef _SCATTER_TRT_H_
#define _SCATTER_TRT_H_

#include <string>
#include <NvInferPlugin.h>

#include "common/reduction.h"

namespace tensorrt_scatter
{
namespace plugin
{

class ScatterPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    ScatterPlugin() = delete;
    ScatterPlugin(int32_t dim, size_t dimSize, ReductionType reduce);
    ScatterPlugin(const void* data, size_t length);

    void setWithBase(bool withBase) noexcept;
    bool getWithBase() const noexcept;

    // IPluginV2 Methods
    const nvinfer1::AsciiChar* getPluginType() const noexcept override;
    const nvinfer1::AsciiChar* getPluginVersion() const noexcept override;
    int32_t getNbOutputs() const noexcept override;
    int32_t initialize() noexcept override;
    void terminate() noexcept override;
    size_t getSerializationSize() const noexcept override;
    void serialize(void* buffer) const noexcept override;
    void destroy() noexcept override;
    void setPluginNamespace(const char* pluginNamespace) noexcept override;
    const nvinfer1::AsciiChar* getPluginNamespace() const noexcept override;
    
    // IPluginV2Ext Methods
    nvinfer1::DataType getOutputDataType(int32_t index, const nvinfer1::DataType* inputTypes, 
        int32_t nbInputs) const noexcept override;
    
    // IPluginV2DynamicExt Methods
    nvinfer1::IPluginV2DynamicExt* clone() const noexcept override;
    nvinfer1::DimsExprs getOutputDimensions(
        int32_t outputIndex, const nvinfer1::DimsExprs* inputs, int32_t nbInputs, nvinfer1::IExprBuilder& exprBuilder) noexcept override;
    bool supportsFormatCombination(
        int32_t pos, const nvinfer1::PluginTensorDesc* inOut, int32_t nbInputs, int32_t nbOutputs) noexcept override;
    void configurePlugin(
        const nvinfer1::DynamicPluginTensorDesc* in, int32_t nbInputs, const nvinfer1::DynamicPluginTensorDesc* out, int32_t nbOutputs) noexcept override;
    size_t getWorkspaceSize(
        const nvinfer1::PluginTensorDesc* inputs, int32_t nbInputs, const nvinfer1::PluginTensorDesc* outputs, int32_t nbOutputs) const noexcept override;
    int32_t enqueue(
        const nvinfer1::PluginTensorDesc* inputDesc, const nvinfer1::PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, 
        void* workspace, cudaStream_t stream) noexcept override;

private:
    std::string mPluginNamespace;
    int32_t mDim;
    size_t mDimSize;
    ReductionType mReduce;
    bool mWithBase;
};

class ScatterPluginCreator : public nvinfer1::IPluginCreator
{
public:
    ScatterPluginCreator();
    const nvinfer1::AsciiChar* getPluginName() const noexcept override;
    const nvinfer1::AsciiChar* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const nvinfer1::AsciiChar* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const nvinfer1::AsciiChar* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const nvinfer1::AsciiChar* pluginNamespace) noexcept override;
    const nvinfer1::AsciiChar* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    static std::vector<nvinfer1::PluginField> mPluginAttributes;
    std::string mPluginNamespace;
};

} // namespace plugin
} // namespace torch_scatter

#endif // _SCATTER_MAX_TRT_H_