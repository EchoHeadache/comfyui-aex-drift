import { app } from "../../scripts/app.js";

// AEX Drift Check's report goes in the frontend's own text preview, the one Preview Any shows
app.registerExtension({
    name: "AEXDrift.Check",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "AEXDriftCheck") return;
        const { addTextPreviewWidgets, updateTextPreviewWidgets } = window.comfyAPI.textPreviewWidgets;
        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            onNodeCreated?.apply(this, arguments);
            addTextPreviewWidgets(this);
        };
        const onExecuted = nodeType.prototype.onExecuted;
        nodeType.prototype.onExecuted = function (message) {
            onExecuted?.apply(this, arguments);
            updateTextPreviewWidgets(this, message);
        };
    },
});
