VisoMaster Fusion Mac launcher

This launcher points to your current installation at /Applications/visomaster-fusion
and uses /opt/miniconda3/envs/visomaster-mac/bin/python. It does not install,
replace, or update the application, Python environment, or any model file.

1. Extract this ZIP.
2. Move "VisoMaster Fusion.app" to your Applications folder (or ~/Applications).
3. Double-click the app to start VisoMaster. The app window may take a while to
   appear when CoreML compiles models for the first time.

The launcher enables the same four approved settings you currently use:
VISOMASTER_COREML_STATIC, VISOMASTER_COREML_GPU_ONLY,
VISOMASTER_INSWAPPER_OPTIMIZED, and VISOMASTER_ALPHAFACE_OPTIMIZED.

If the app cannot start, an alert appears and details are written to
~/Library/Logs/VisoMasterFusion/launcher.log.

"VisoMaster Fusion.command" is an optional Terminal launcher. It uses exactly
the same settings and shows the app's logs live. You can keep it anywhere.

If macOS asks for confirmation when first opening a downloaded launcher,
Control-click the app and choose Open.
