
import omni.kit.test
import omni.usd
import carb
from omni.kit.viewport.utility.tests import capture_viewport_and_compare

import sys
import unittest

from pathlib import Path


EXTENSION_ROOT = Path(carb.tokens.get_tokens_interface().resolve("${omni.hydra.pxr}")).resolve().absolute()
TESTS_ROOT = EXTENSION_ROOT.joinpath("data", "tests")
USD_SCENES = TESTS_ROOT.joinpath("usd")
GOLDEN_IMAGES = TESTS_ROOT.joinpath("images")
OUTPUTS_DIR = Path(omni.kit.test.get_test_output_path())


class HydraTextureTest(omni.kit.test.AsyncTestCase):
    async def setUp(self):
        self.usd_context = omni.usd.get_context()
        await self.usd_context.new_stage_async()

    async def tearDown(self):
        await self.usd_context.new_stage_async()

    async def run_imge_test(self, usd_file: str, image_name: str, fn = None, wait_iterations: int = 5, threshold: int = 10):
        await self.usd_context.open_stage_async(str(USD_SCENES.joinpath(usd_file)))

        app = omni.kit.app.get_app()
        for i in range(wait_iterations):
            await app.next_update_async()

        if fn:
            fn()

        for i in range(wait_iterations):
            await app.next_update_async()

        passed, fail_msg = await capture_viewport_and_compare(image_name, threshold=threshold, output_img_dir=OUTPUTS_DIR, golden_img_dir=GOLDEN_IMAGES)
        self.assertTrue(passed, msg=fail_msg)

    async def test_render_cube(self):
        """Test rendering produces an image"""
        await self.run_imge_test('cube.usda', 'render_cube.png')

    async def test_render_cubes(self):
        """Test rendering produces an image after re-open"""
        await self.run_imge_test('cubes.usda', 'render_cubes.png')

    async def test_select_cube_x(self):
        """Test selection on cube_x highlights correct prim"""
        await self.run_imge_test('cubes.usda', 'select_cube_x.png',
                                 lambda: self.usd_context.get_selection().set_selected_prim_paths(['/World/cube_x'], True)
        )

    async def test_select_cube_y(self):
        """Test selection on cube_y highlights correct prim"""
        await self.run_imge_test('cubes.usda', 'select_cube_y.png',
                                 lambda: self.usd_context.get_selection().set_selected_prim_paths(['/World/cube_y'], True)
        )

    async def test_select_cube_z(self):
        """Test selection on cube_z highlights correct prim"""
        await self.run_imge_test('cubes.usda', 'select_cube_z.png',
                                 lambda: self.usd_context.get_selection().set_selected_prim_paths(['/World/cube_z'], True)
        )

    async def test_display_purpose(self):
        """Test the carb display purpose settings"""

        def set_puposes(guide: bool, proxy: bool, render: bool):
            settings = carb.settings.get_settings()
            settings.set('/persistent/app/hydra/displayPurpose/guide', guide)
            settings.set('/persistent/app/hydra/displayPurpose/proxy', proxy)
            settings.set('/persistent/app/hydra/displayPurpose/render', render)

        try:
            # Test, guide, proxy, and render purposes render by default
            await self.run_imge_test('purpose.usda', 'all_purposes.png')

            # Test with proxy purpose disabled
            set_puposes(False, True, True)
            await self.run_imge_test('purpose.usda', 'proxy_purposes.png')

            # Test with guide purpose disabled
            set_puposes(True, False, True)
            await self.run_imge_test('purpose.usda', 'guide_purposes.png')

            # Test with render purpose disabled
            set_puposes(True, True, False)
            await self.run_imge_test('purpose.usda', 'render_purposes.png')
        finally:
            set_puposes(True, True, True)

        # Run the all purpose test once more now that all have been re-enabled
        await self.run_imge_test('purpose.usda', 'all_purposes.png')

    async def test_rtx_render_modes(self):
        """Test some RTX render settings are honored"""

        settings = carb.settings.get_settings()
        settings.set("/rtx/debugMaterialType", -1)
        settings.set("/rtx/wireframe/mode", 0)
        try:
            # Test, guide, proxy, and render purposes render by default
            await self.run_imge_test('render_modes.usda', 'render_modes_default.png')

            settings.set("/rtx/debugMaterialType", 0)
            await self.run_imge_test('render_modes.usda', 'render_modes_no_material.png')

            # Tests below create images that are outside a meaningful range/image test on Windows with Vulkan
            # Since they are highly specific test of wireframe rendering, just skip them in that case
            import sys
            if sys.platform.startswith('win') and settings.get_as_bool("/app/vulkan"):
                return

            settings.set("/rtx/wireframe/mode", 2)
            await self.run_imge_test('render_modes.usda', 'render_modes_no_material_wire.png')

            settings.set("/rtx/debugMaterialType", -1)
            await self.run_imge_test('render_modes.usda', 'render_modes_wire_material.png')
        finally:
            settings.set("/rtx/debugMaterialType", -1)
            settings.set("/rtx/wireframe/mode", 0)
