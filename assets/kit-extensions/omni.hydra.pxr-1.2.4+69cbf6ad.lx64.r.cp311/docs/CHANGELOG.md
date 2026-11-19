# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2.4] - 2024-07-02
### Added
-  OM-124514: Added setting to avoid starting with debug materials enabled.

## [1.2.3] - 2024-05-01
### Added
-  OM-124514: Added setting to avoid starting with debug materials enabled.

## [1.2.2] - 2024-01-12
### Changed
- Allow extension to be toggled on, but not unloaded.

## [1.2.1] - 2023-12-27
### Fixed
- Possible nullptr dereferencing when FSD is enabled.
- Lack of selection hilighting when FSD is enabled.
- Use HdxTaskController::SetSelectionLocateColor when available.
### Added
- Warning when window policy is set to something other than horizontal matching when FSD is enabled.
### Removed
- Unnecessary dependencies.

## [1.2.0] - 2023-12-12
### Added
- Support for multiple HdRenderDelegates (from alternate UsdContexts)
- Ability for extension to function in true headless/no-ui mode.

## [1.1.5] - 2023-10-31
### Fixed
- Make sure to release UsdStage and event handlers when moving to a no-stage state (on shutdown).

## [1.1.4] - 2023-08-10
### Changed
- Remove gl_VertexID usage from GLSL to mitigate intel GPU compatibility issues.
### Added
- Run tests in compatibility mode with Vulkan.

## [1.1.3] - 2023-02-13
### Fixed
- Proper omni.usd.core dependency for loading extension in isolation.

## [1.1.2] - 2022-09-12
### Added
- Support for render purpose via carb settings.
- Live update of selection color changes.
- Support for RTX render modes.

## [1.1.1] - 2022-06-06
### Added
- Support for /app/viewport/outline/enabled setting

## [1.1.0] - 2021-09-06
### Added
- OM-35722: Enable omni.hydra.pxr for kit-dev app
- OM-35753: Support for Render Settings via Hydra API
- OM-35747: Fix uninitialized variables leading to OpenGL errors on first draw.
- OM-35745: Allow user to specify delegate to start  with
- OM-35756: Use Kit's selection color and outline width
- OM-35756: Enable marquee selection in Kit viewport
- OM-35753: Fix leaks in picking and selection transfer
- Fix issue of creating instance of Storm before requested delegate

## [1.0.0] - 2020-11-05
### Added
- Initial implementation of Hydra render engine in Kit
