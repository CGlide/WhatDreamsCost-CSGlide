import { app } from "../../scripts/app.js";

// LTX Director Guide CS is a pure pass-through processor node.
// All configuration (images, insert frames, strengths) comes from
// the guide_data output of the LTX Director CS (Timeline) node.
app.registerExtension({
    name: "Comfy.LTXDirectorGuideCS",
    async nodeCreated(node) {
        if (node.comfyClass !== "LTXDirectorGuideCS") return;
        // Nothing to initialize — the node has no configurable widgets.
    },
});
