#ifndef TRTS_GATHER_CSR_PLUGIN_H
#define TRTS_GATHER_CSR_PLUGIN_H

#include <string>
#include <NvInferPlugin.h>

namespace tensorrt_scatter
{
namespace plugin
{

class GatherCSRPlugin : public nvinfer1::IPluginV2DynamicExt
{
public:
    GatherCSRPlugin();
    GatherCSRPlugin(const void* data, size_t length);

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
};

class GatherCSRPluginCreator : public nvinfer1::IPluginCreator
{
public:
    GatherCSRPluginCreator();
    const nvinfer1::AsciiChar* getPluginName() const noexcept override;
    const nvinfer1::AsciiChar* getPluginVersion() const noexcept override;
    const nvinfer1::PluginFieldCollection* getFieldNames() noexcept override;
    nvinfer1::IPluginV2* createPlugin(const nvinfer1::AsciiChar* name, const nvinfer1::PluginFieldCollection* fc) noexcept override;
    nvinfer1::IPluginV2* deserializePlugin(const nvinfer1::AsciiChar* name, const void* serialData, size_t serialLength) noexcept override;
    void setPluginNamespace(const nvinfer1::AsciiChar* pluginNamespace) noexcept override;
    const nvinfer1::AsciiChar* getPluginNamespace() const noexcept override;

private:
    static nvinfer1::PluginFieldCollection mFC;
    std::string mPluginNamespace;
};

} // namespace plugin
} // namespace torch_scatter

#endif // TRTS_GATHER_CSR_PLUGIN_H