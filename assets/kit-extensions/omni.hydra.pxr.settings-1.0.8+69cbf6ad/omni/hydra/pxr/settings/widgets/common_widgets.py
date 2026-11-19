import omni.ui as ui
import carb

from omni.rtx.window.settings.rtx_settings_stack import RTXSettingsStack
from omni.rtx.window.settings.settings_collection_frame import SettingsCollectionFrame
from omni.kit.widget.settings import SettingType


class CarbDictSettingsFrame(SettingsCollectionFrame):
    def __value_to_type(self, value):
        if isinstance(value, bool):
            return SettingType.BOOL
        if isinstance(value, int):
            return SettingType.INT
        if isinstance(value, float):
            return SettingType.FLOAT
        if isinstance(value, str):
            return SettingType.STRING
        if isinstance(value, carb.ColorRgb):
            return SettingType.COLOR3
        if isinstance(value, carb.Double3):
            return SettingType.DOUBLE3
        if isinstance(value, carb.Double2):
            return SettingType.DOUBLE2
        if isinstance(value, carb.Int2):
            return SettingType.INT2

    def __init__(self, desc, *args, **kwargs):
        super().__init__(desc.display_name, *args, **kwargs)
        self._settings_key = desc.settings_key
        self._plugin_id = desc.plugin_id

    def _add_common_settings(self):
        # Add the renderer-specific light adjustment
        self._add_setting(SettingType.FLOAT, name='Renderer Lighting Factor',
                        path=f'/persistent/app/hydra/delegates/{self._plugin_id}/omniLightAdjustment',
                        tooltip='Renderer specific amount to scale lighting by.')

    def _add_setting_dictionary(self, setting_dict: dict, root_key: str, sub_dicts: dict):
        for name, setting in setting_dict.items():
            value = setting.get('value', None)
            if value is None:
                if isinstance(setting, dict):
                    sub_dicts[f'{root_key}/{name}'] = setting
                continue
            setting_type = self.__value_to_type(value)
            if setting_type is not None:
                value_key = f'{root_key}/{name}/value'
                self._add_setting(setting_type, name=setting.get('name', name), path=value_key, tooltip=setting.get('tooltip', ''))

    def _build_ui(self):
        # Add all the common renderer specific settings
        self._add_common_settings()

        settings = carb.settings.get_settings().get(self._settings_key)
        if settings:
            # Collect all namespaced entries, and do them last
            namespaced = {}
            # Start with the root item
            self._add_setting_dictionary(settings, self._settings_key, namespaced)
            # Error/Exit if over 100 nested dictionaries / namespaces to avoid infinite loop
            ns_counter = 100
            while namespaced:
                # Swap to a local_dict and re-write into namespaced
                local_dict, namespaced = namespaced, {}
                for root_key, root_dict in local_dict.items():
                    self._add_setting_dictionary(root_dict, root_key, namespaced)

                ns_counter = ns_counter - 1
                if ns_counter <= 0:
                    carb.log_error('Exiting CarbDictSettingsFrame because of unexpectedly large number of dicitonaries')
                    break


class PxrHydraRendererSettingStack(RTXSettingsStack):
    def __init__(self, desc, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # _self._stack is required for RTXSettingsStack
        self._stack = ui.VStack(spacing=10)
        with self._stack:
            CarbDictSettingsFrame(desc, parent=self)
            ui.Spacer()


class GlobalSettingsFrame(SettingsCollectionFrame):
    def _build_ui(self):
        # Add the global light adjustment setting
        self._add_setting(SettingType.FLOAT, name='Global Lighting Factor',
                          path=f'/persistent/app/hydra/delegates/common/omniGlobalLightAdjustment',
                          tooltip='Global amount to scale lighting by.')


class CommonPxrHydraSettingStack(RTXSettingsStack):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        # _self._stack is required for RTXSettingsStack
        self._stack = ui.VStack(spacing=10)
        with self._stack:
            GlobalSettingsFrame('Global', parent=self)
            ui.Spacer()
