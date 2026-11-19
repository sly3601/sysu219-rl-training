__all__ = ['RendererSettingsFactory', 'PxrHydraSettingsExtension']

from .extension import *


# Expose our own RendererSettingsFactory instead of having users use the RTX version directly.
# We'll just call into the RTX-RendererSettingsFactory for now, but add a few methods to add and
# remove the renderer from the Viewport's available list.
from omni.rtx.window.settings import RendererSettingsFactory as RTXRendererSettingsFactory
from typing import List
import carb


class RendererSettingsFactory(RTXRendererSettingsFactory):
    @staticmethod
    def register_renderer(plugin_id: str, stacks_list: List[str], display_name: str):
        """Register a list of stacks for visibility in the RenderSettings Widget.
        This will also let the renderer be available in the Viewport renderer drop-down.
        """
        RTXRendererSettingsFactory.register_renderer(plugin_id, stacks_list)
        RendererSettingsFactory.enable_viewport_renderer(plugin_id, display_name)

    @staticmethod
    def unregister_renderer(plugin_id: str):
        """De-register a renderer in the RenderSettings Widget.
        This will also remove the renderer from the Viewport renderer drop-down.
        """
        RTXRendererSettingsFactory.unregister_renderer(plugin_id)
        RendererSettingsFactory.disable_viewport_renderer(plugin_id)

    __g_renderers = {}

    @staticmethod
    def enable_viewport_renderer(plugin_id: str, display_name: str):
        # Subject to change, but currently this is needed for Viewport-1's render-menu
        RendererSettingsFactory.__g_renderers[plugin_id] = display_name
        settings = carb.settings.get_settings()
        renderers = settings.get('/pxr/renderers')
        cur_key = f'{plugin_id}:{display_name}'
        if not renderers or renderers.find(cur_key) == -1:
            renderers = (renderers + '|') if renderers else ''
            settings.set('/pxr/renderers', renderers + cur_key)

    @staticmethod
    def disable_viewport_renderer(plugin_id: str):
        cur_key = f'{plugin_id}:{RendererSettingsFactory.__g_renderers[plugin_id]}'
        del RendererSettingsFactory.__g_renderers[plugin_id]
        # Again, subject to change, but remove ourself from the Viewport-1 menu
        # We don't know where we were inserted so just be careful and remove any empty spots we may create
        settings = carb.settings.get_settings()
        renderers = settings.get('/pxr/renderers')
        if not renderers:
            return
        renderers = renderers.replace(cur_key, '').replace('||', '')
        if renderers.startswith('|'):
            renderers = renderers[1:]
        if renderers.endswith('|'):
            renderers = renderers[0:-1]
        settings.set('/pxr/renderers', renderers)
