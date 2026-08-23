#include <EGL/egl.h>
#include <EGL/eglext.h>
#include <stdio.h>

static int initialize_display(EGLDisplay display) {
    EGLint major = 0;
    EGLint minor = 0;

    if (display == EGL_NO_DISPLAY || !eglInitialize(display, &major, &minor)) {
        return 0;
    }

    const char *vendor = eglQueryString(display, EGL_VENDOR);
    const char *version = eglQueryString(display, EGL_VERSION);
    printf("EGL initialized: version=%d.%d vendor=%s api=%s\n", major, minor,
           vendor ? vendor : "unknown", version ? version : "unknown");
    eglTerminate(display);
    return 1;
}

int main(void) {
    PFNEGLQUERYDEVICESEXTPROC query_devices =
        (PFNEGLQUERYDEVICESEXTPROC)eglGetProcAddress("eglQueryDevicesEXT");
    PFNEGLGETPLATFORMDISPLAYEXTPROC get_platform_display =
        (PFNEGLGETPLATFORMDISPLAYEXTPROC)eglGetProcAddress("eglGetPlatformDisplayEXT");

    if (query_devices && get_platform_display) {
        EGLDeviceEXT devices[32];
        EGLint count = 0;
        if (query_devices(32, devices, &count)) {
            for (EGLint index = 0; index < count; ++index) {
                EGLDisplay display = get_platform_display(
                    EGL_PLATFORM_DEVICE_EXT, devices[index], NULL);
                if (initialize_display(display)) {
                    printf("EGL device index: %d of %d\n", index, count);
                    return 0;
                }
            }
        }
    }

    if (initialize_display(eglGetDisplay(EGL_DEFAULT_DISPLAY))) {
        return 0;
    }

    fprintf(stderr, "unable to initialize EGL on any device\n");
    return 1;
}
