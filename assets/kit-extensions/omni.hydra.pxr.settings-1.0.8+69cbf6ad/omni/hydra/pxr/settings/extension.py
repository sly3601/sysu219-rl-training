import omni.ext
import omni.kit.app
import carb.settings

from omni.rtx.window.settings import RendererSettingsFactory
from .widgets.common_widgets import CommonPxrHydraSettingStack, PxrHydraRendererSettingStack

kLoadedDelegatesSettingsPath = '/pxr/renderers'
kActiveDelegatesSettingsPath = '/pxr/rendermode'
kActiveHydraEnginePath = '/renderer/active'
kSavedDelegatesSettingsPath = '/persistent/app/hydra/delegates'
kDelegateSettingsKey = 'settings'


class HdSettingsDesc(object):
    def __init__(self, *args):
        self.plugin_id, self.display_name, self.settings_key = args


def _build_settings(settings, instatiator):
    instances = []
    delegates = settings.get(kLoadedDelegatesSettingsPath)
    delegates = [names.split(':') for names in delegates.split('|')] if delegates else []
    # for plugin_name, config_dict in settings.get(kAllDelegatesSettingsPath).items():
    for plugin_name, display_name in delegates:
        if display_name == 'GL':
            display_name = 'Storm'
        settings_key = kSavedDelegatesSettingsPath + '/' + plugin_name + '/' + kDelegateSettingsKey
        instance = instatiator(HdSettingsDesc(plugin_name, display_name, settings_key))
        if instance:
            instances.append(instance)

    return instances


class PxrHydraSettingsExtension(omni.ext.IExt):
    """The entry point for External Delegate Settings Window"""

    def __renderersUpdated(self, loaded_delegates_value, event_type):
        if event_type != carb.settings.ChangeEventType.CHANGED:
            return

        # Delay all creation until omni.hydra.pxr is ctive and using a loaded delegate
        settings = carb.settings.get_settings()
        if settings.get(kActiveHydraEnginePath) != 'pxr':
            return
        active_plugin = settings.get(kActiveDelegatesSettingsPath)
        if not active_plugin:
            return

        def buildRendererSettings(desc):
            plugin_id = desc.plugin_id
            # Check if an  extension has already registered a Render Settings widget for this renderer.
            rs = RendererSettingsFactory._get_render_settings_extension()
            if rs and plugin_id in rs._settings_window.get_registered_renderers():
                return
            if active_plugin == plugin_id and not plugin_id in self.__hdsettings:
                RendererSettingsFactory.register_renderer(desc.plugin_id, ['Global', desc.display_name])
                RendererSettingsFactory.register_stack('Global',
                    lambda desc = desc: CommonPxrHydraSettingStack())
                RendererSettingsFactory.register_stack(desc.display_name,
                    lambda desc = desc: PxrHydraRendererSettingStack(desc))
                return desc

        added = _build_settings(settings, buildRendererSettings)
        for desc in added:
            self.__hdsettings.append(desc.plugin_id)

    def on_startup(self):
        self.__hdsettings = []
        # XXX: Need a better and more common way to watch for renderer-changes
        self.__hd_renderers_changed = (
            omni.kit.app.SettingChangeSubscription(kLoadedDelegatesSettingsPath, self.__renderersUpdated),
            omni.kit.app.SettingChangeSubscription(kActiveHydraEnginePath, self.__renderersUpdated),
            omni.kit.app.SettingChangeSubscription(kActiveDelegatesSettingsPath, self.__renderersUpdated)
        )

    def on_shutdown(self):
        descs = self.__hdsettings
        self.__hdsettings = None
        self.__hd_renderers_changed = None
        for plugin_id in descs:
            RendererSettingsFactory.unregister_renderer(plugin_id)
            RendererSettingsFactory.unregister_stack(plugin_id)
        RendererSettingsFactory.build_ui()
