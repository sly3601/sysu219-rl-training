import omni.kit.test
import omni.kit.app
from omni.hydra.pxr.settings import RendererSettingsFactory
from omni.kit.viewport.utility import get_active_viewport_window

from omni.hydra.pxr.settings.widgets.common_widgets import CommonPxrHydraSettingStack, PxrHydraRendererSettingStack
from omni.hydra.pxr.settings.extension import HdSettingsDesc, kSavedDelegatesSettingsPath, kDelegateSettingsKey

from omni.rtx.window.settings.tests.test_ui import TestSetRenderSettings
from omni.rtx.window.settings import RendererSettingsFactory as RTXRendererSettingsFactory
from omni.rtx.window.settings.rtx_settings_stack import RTXSettingsStack
from omni.rtx.window.settings.settings_collection_frame import FrameSessionState

from omni.ui.tests.test_base import OmniUiTest

from pathlib import Path
import carb
import omni.ui

DATA_PATH = Path(carb.tokens.get_tokens_interface().resolve("${omni.hydra.pxr.settings}/data")).absolute().resolve()
GOLDEN_IMAGE_DIR = DATA_PATH.joinpath("tests").absolute().resolve()


class TestPXRCommandsDefaults(TestSetRenderSettings, OmniUiTest):
    async def setUp(self):
        # Load the scene
        # This currently needs to happen as omni.hydra.pxr settings are only known at runtime,
        # once the renderer is loaded, and it will only be loaded when a stage is attached.
        scene_path = str(DATA_PATH.joinpath("tests", "cube.usda"))
        await omni.usd.get_context().open_stage_async(scene_path)
        await omni.kit.app.get_app().next_update_async()

    async def test_pxr_settings_ui(self):
        # Clear this out, so both legacy and viewport-next start at the same place
        carb.settings.get_settings().set('/pxr/renderers', '')
        # Enable Storm for some tests in run_test_render_settings_ui
        RendererSettingsFactory.enable_viewport_renderer('HdStormRendererPlugin', 'Storm')

        # Make sure the renderer was registered in the UI stack
        renderers = RendererSettingsFactory.get_registered_renderers()
        self.assertNotEqual(len(renderers), 0)

        # Run the base-line render-settings window ui tests on the known settings path
        await self.run_test_render_settings_ui("/persistent/app/hydra/delegates/HdStormRendererPlugin/settings")
        RendererSettingsFactory.disable_viewport_renderer('HdStormRendererPlugin')

    async def test_hd_storm_unload(self):
        # Simulate a load and make sure it appears in the renderer-list
        RendererSettingsFactory.enable_viewport_renderer('HdStormRendererPlugin', 'Storm')
        self.assertNotEqual(carb.settings.get_settings().get("/pxr/renderers").find("HdStormRendererPlugin:Storm"), -1)

        # Simulate an unload and make sure it no longer appears in the renderer-list
        RendererSettingsFactory.disable_viewport_renderer('HdStormRendererPlugin')
        self.assertEqual(carb.settings.get_settings().get("/pxr/renderers").find("HdStormRendererPlugin:Storm"), -1)

    async def test_carb_dict_to_settings(self):
        WIDTH, HEIGHT = 640, 500
        await self.create_test_area(width=WIDTH, height=HEIGHT)

        # Build a fake plugin to test
        test_plugin_id = 'HdRenderPluginDictTest'
        # This is where settings are stored
        settings_key = f'{kSavedDelegatesSettingsPath}/{test_plugin_id}/{kDelegateSettingsKey}'
        # Simple object that is used to pass to build settings
        test_plugin_desc = HdSettingsDesc(test_plugin_id, 'DictTest', settings_key)
        # Generate a few settings, importantly to test nested-dictionary -> USD namespace conversion
        settings_dict = {
            'namespace1': {
                'setting1': {
                    'name': 'namespace1_LABEL',
                    'value': 'namespace1_VALUE'
                },
                'namespace2': {
                    'setting2': {
                        'name': 'namespace2_LABEL',
                        'value': True
                    },
                    'namespace3': {
                        'setting3': {
                            'name': 'namespace3_LABEL',
                            'value': 3.0
                        }
                    }
                }
            }
        }
        # Set the defaults
        settings = carb.settings.get_settings()
        settings.set(settings_key, settings_dict)

        # Setup a default state that won't cause issues with test
        # Expand the the render-settings for capture
        FrameSessionState.set_state('omni.hydra.pxr.settings.widgets.common_widgets.CarbDictSettingsFrame', 'collapsed', False)
        # omni.ui conversion error if the value is initially None
        settings.set(f'/persistent/app/hydra/delegates/{test_plugin_id}/omniLightAdjustment', 1.0)

        # XXX: RendererSettings needs this to be set properly for Viewport-2 case, but not Viewport-1
        class ActiveRendererRIIA:
            def __init__(self):
                carb.settings.get_settings().set('/pxr/rendermode', 'HdRenderPluginDictTest')
            def __del__(self):
                carb.settings.get_settings().set('/pxr/rendermode', 'HdStormRendererPlugin')
        active_raii = ActiveRendererRIIA()

        # Register all the settings stacks
        RTXRendererSettingsFactory.register_renderer(test_plugin_id, ['Global', test_plugin_desc.display_name])
        RTXRendererSettingsFactory.register_stack('Global', lambda *args: CommonPxrHydraSettingStack())
        RTXRendererSettingsFactory.register_stack(test_plugin_desc.display_name, lambda *args: PxrHydraRendererSettingStack(test_plugin_desc))
        RTXRendererSettingsFactory.set_current_stack(test_plugin_desc.display_name)

        # Get a reference to the window and resize it
        window = omni.ui.Window(title='Render Settings')
        window.width = WIDTH
        window.height = HEIGHT

        # Hide the Viewport window from the screen-shot
        # We're currently dependent on it to load omni.hydra.pxr and Storm's runtime settings
        viewport_window = get_active_viewport_window()
        if viewport_window:
            viewport_window.visible = False

        for i in range(3):
            await omni.kit.app.get_app().next_update_async()

        # Unegister everything
        RTXRendererSettingsFactory.unregister_renderer(test_plugin_id)
        RTXRendererSettingsFactory.unregister_stack(test_plugin_id)

        await self.finalize_test(golden_img_dir=GOLDEN_IMAGE_DIR, golden_img_name='test_carb_dict_to_settings.png')
