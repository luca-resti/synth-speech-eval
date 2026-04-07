pyarmor gen -O dist --platform windows.x86_64 --platform linux.x86_64 -r app/
rmdir /s /q "pyarmor_runtime_000000"
move /y "dist/pyarmor_runtime_000000" "pyarmor_runtime_000000"
docker build -t synth-speech-eval .