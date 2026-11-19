# Public API for module omni.hydra.pxr.settings:

## Classes

- class RendererSettingsFactory(RTXRendererSettingsFactory)
  - static def register_renderer(plugin_id: str, stacks_list: List[str], display_name: str)
  - static def unregister_renderer(plugin_id: str)
  - static def enable_viewport_renderer(plugin_id: str, display_name: str)
  - static def disable_viewport_renderer(plugin_id: str)

- class PxrHydraSettingsExtension(omni.ext.IExt)
  - def on_startup(self)
  - def on_shutdown(self)
