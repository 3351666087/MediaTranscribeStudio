#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <objc/runtime.h>

#include <cstdio>
#include <vector>

namespace {

struct ClassBridgeState {
    Class cls = Nil;
    IMP operating_system_version_imp = nullptr;
    IMP is_at_least_imp = nullptr;
    IMP operating_system_version_string_imp = nullptr;
    const char* operating_system_version_types = nullptr;
    const char* is_at_least_types = nullptr;
    const char* operating_system_version_string_types = nullptr;
};

struct BridgeState {
    bool installed = false;
    bool future_only = true;
    bool debug_logging = false;
    NSOperatingSystemVersion target = {15, 0, 0};
    std::vector<ClassBridgeState> classes;
};

BridgeState& state() {
    static BridgeState s;
    return s;
}

using OperatingSystemVersionFn = NSOperatingSystemVersion (*)(id, SEL);
using IsAtLeastFn = BOOL (*)(id, SEL, NSOperatingSystemVersion);
using OperatingSystemVersionStringFn = NSString* (*)(id, SEL);

bool version_greater(
    const NSOperatingSystemVersion& lhs,
    const NSOperatingSystemVersion& rhs
) {
    if (lhs.majorVersion != rhs.majorVersion) {
        return lhs.majorVersion > rhs.majorVersion;
    }
    if (lhs.minorVersion != rhs.minorVersion) {
        return lhs.minorVersion > rhs.minorVersion;
    }
    return lhs.patchVersion > rhs.patchVersion;
}

bool version_at_least(
    const NSOperatingSystemVersion& current,
    const NSOperatingSystemVersion& required
) {
    if (current.majorVersion != required.majorVersion) {
        return current.majorVersion > required.majorVersion;
    }
    if (current.minorVersion != required.minorVersion) {
        return current.minorVersion > required.minorVersion;
    }
    return current.patchVersion >= required.patchVersion;
}

NSOperatingSystemVersion bridged_version(const NSOperatingSystemVersion& actual) {
    const BridgeState& bridge = state();
    if (!bridge.installed) {
        return actual;
    }
    if (!bridge.future_only) {
        return bridge.target;
    }
    if (version_greater(actual, bridge.target)) {
        return bridge.target;
    }
    return actual;
}

bool versions_equal(
    const NSOperatingSystemVersion& lhs,
    const NSOperatingSystemVersion& rhs
) {
    return lhs.majorVersion == rhs.majorVersion
        && lhs.minorVersion == rhs.minorVersion
        && lhs.patchVersion == rhs.patchVersion;
}

NSString* formatted_version_string(const NSOperatingSystemVersion& value) {
    return [NSString stringWithFormat:@"%ld.%ld.%ld",
            static_cast<long>(value.majorVersion),
            static_cast<long>(value.minorVersion),
            static_cast<long>(value.patchVersion)];
}

void maybe_log_install(
    const NSOperatingSystemVersion& actual,
    const NSOperatingSystemVersion& effective
) {
    const BridgeState& bridge = state();
    if (!bridge.debug_logging) {
        return;
    }
    std::fprintf(
        stderr,
        "[mts_apple_runtime_shim] NSProcessInfo bridge active: %ld.%ld.%ld -> %ld.%ld.%ld\n",
        static_cast<long>(actual.majorVersion),
        static_cast<long>(actual.minorVersion),
        static_cast<long>(actual.patchVersion),
        static_cast<long>(effective.majorVersion),
        static_cast<long>(effective.minorVersion),
        static_cast<long>(effective.patchVersion)
    );
    std::fflush(stderr);
}

ClassBridgeState* find_class_state_for_object(id self) {
    Class runtime_cls = object_getClass(self);
    for (auto& item : state().classes) {
        if (item.cls == runtime_cls) {
            return &item;
        }
    }
    for (auto& item : state().classes) {
        if (item.cls == [NSProcessInfo class]) {
            return &item;
        }
    }
    return nullptr;
}

NSOperatingSystemVersion actual_version_for_object(id self, SEL cmd) {
    ClassBridgeState* cls_state = find_class_state_for_object(self);
    if (cls_state == nullptr || cls_state->operating_system_version_imp == nullptr) {
        return {0, 0, 0};
    }
    auto original = reinterpret_cast<OperatingSystemVersionFn>(cls_state->operating_system_version_imp);
    const NSOperatingSystemVersion actual = original(self, cmd);
    return bridged_version(actual);
}

NSOperatingSystemVersion actual_unbridged_version_for_object(id self) {
    ClassBridgeState* cls_state = find_class_state_for_object(self);
    if (cls_state == nullptr || cls_state->operating_system_version_imp == nullptr) {
        return {0, 0, 0};
    }
    auto original = reinterpret_cast<OperatingSystemVersionFn>(cls_state->operating_system_version_imp);
    return original(self, @selector(operatingSystemVersion));
}

NSOperatingSystemVersion mts_operating_system_version(id self, SEL cmd) {
    return actual_version_for_object(self, cmd);
}

BOOL mts_is_operating_system_at_least_version(
    id self,
    SEL cmd,
    NSOperatingSystemVersion required
) {
    ClassBridgeState* cls_state = find_class_state_for_object(self);
    if (cls_state == nullptr) {
        return NO;
    }
    if (cls_state->operating_system_version_imp == nullptr) {
        auto original_check = reinterpret_cast<IsAtLeastFn>(cls_state->is_at_least_imp);
        return original_check == nullptr ? NO : original_check(self, cmd, required);
    }
    auto original = reinterpret_cast<OperatingSystemVersionFn>(cls_state->operating_system_version_imp);
    const NSOperatingSystemVersion actual = original(self, @selector(operatingSystemVersion));
    const NSOperatingSystemVersion effective = bridged_version(actual);
    return version_at_least(effective, required) ? YES : NO;
}

NSString* mts_operating_system_version_string(id self, SEL cmd) {
    ClassBridgeState* cls_state = find_class_state_for_object(self);
    NSString* actual_string = nil;
    if (cls_state != nullptr && cls_state->operating_system_version_string_imp != nullptr) {
        auto original = reinterpret_cast<OperatingSystemVersionStringFn>(
            cls_state->operating_system_version_string_imp
        );
        actual_string = original(self, cmd);
    }

    const NSOperatingSystemVersion actual = actual_unbridged_version_for_object(self);
    const NSOperatingSystemVersion effective = bridged_version(actual);
    if (versions_equal(actual, effective)) {
        return actual_string != nil ? actual_string : formatted_version_string(actual);
    }
    return formatted_version_string(effective);
}

bool install_bridge_for_class(Class cls) {
    if (cls == Nil) {
        return false;
    }

    for (const auto& existing : state().classes) {
        if (existing.cls == cls) {
            return true;
        }
    }

    Method version_method = class_getInstanceMethod(cls, @selector(operatingSystemVersion));
    Method at_least_method = class_getInstanceMethod(cls, @selector(isOperatingSystemAtLeastVersion:));
    Method version_string_method = class_getInstanceMethod(cls, @selector(operatingSystemVersionString));
    if (version_method == nullptr || at_least_method == nullptr) {
        return false;
    }

    ClassBridgeState entry;
    entry.cls = cls;
    entry.operating_system_version_imp = class_getMethodImplementation(cls, @selector(operatingSystemVersion));
    entry.is_at_least_imp = class_getMethodImplementation(cls, @selector(isOperatingSystemAtLeastVersion:));
    entry.operating_system_version_string_imp = version_string_method == nullptr
        ? nullptr
        : class_getMethodImplementation(cls, @selector(operatingSystemVersionString));
    entry.operating_system_version_types = method_getTypeEncoding(version_method);
    entry.is_at_least_types = method_getTypeEncoding(at_least_method);
    entry.operating_system_version_string_types = version_string_method == nullptr
        ? nullptr
        : method_getTypeEncoding(version_string_method);
    if (entry.operating_system_version_imp == nullptr || entry.is_at_least_imp == nullptr) {
        return false;
    }

    class_replaceMethod(
        cls,
        @selector(operatingSystemVersion),
        reinterpret_cast<IMP>(mts_operating_system_version),
        entry.operating_system_version_types
    );
    class_replaceMethod(
        cls,
        @selector(isOperatingSystemAtLeastVersion:),
        reinterpret_cast<IMP>(mts_is_operating_system_at_least_version),
        entry.is_at_least_types
    );
    if (
        entry.operating_system_version_string_imp != nullptr
        && entry.operating_system_version_string_types != nullptr
    ) {
        class_replaceMethod(
            cls,
            @selector(operatingSystemVersionString),
            reinterpret_cast<IMP>(mts_operating_system_version_string),
            entry.operating_system_version_string_types
        );
    }

    state().classes.push_back(entry);
    return true;
}

}  // namespace

extern "C" int mts_apple_compat_bridge_install(
    int target_major,
    int target_minor,
    int target_patch,
    int future_only,
    int debug_logging
) {
    @autoreleasepool {
        BridgeState& bridge = state();
        if (bridge.installed) {
            return 1;
        }

        bridge.future_only = future_only != 0;
        bridge.debug_logging = debug_logging != 0;
        bridge.target = {
            static_cast<NSInteger>(target_major > 0 ? target_major : 15),
            static_cast<NSInteger>(target_minor >= 0 ? target_minor : 0),
            static_cast<NSInteger>(target_patch >= 0 ? target_patch : 0),
        };
        bridge.classes.clear();

        Class base_cls = [NSProcessInfo class];
        Class runtime_cls = object_getClass([NSProcessInfo processInfo]);
        bool installed_any = false;
        if (runtime_cls != Nil) {
            installed_any = install_bridge_for_class(runtime_cls) || installed_any;
        }
        if (base_cls != Nil && base_cls != runtime_cls) {
            installed_any = install_bridge_for_class(base_cls) || installed_any;
        }
        if (!installed_any) {
            bridge.classes.clear();
            return 0;
        }

        bridge.installed = true;
        const NSOperatingSystemVersion actual = actual_unbridged_version_for_object(
            [NSProcessInfo processInfo]
        );
        maybe_log_install(actual, bridged_version(actual));
        return 1;
    }
}

extern "C" int mts_apple_compat_bridge_uninstall() {
    @autoreleasepool {
        BridgeState& bridge = state();
        if (!bridge.installed) {
            return 1;
        }

        for (const auto& entry : bridge.classes) {
            if (entry.cls == Nil) {
                continue;
            }
            if (
                entry.operating_system_version_imp != nullptr
                && entry.operating_system_version_types != nullptr
            ) {
                class_replaceMethod(
                    entry.cls,
                    @selector(operatingSystemVersion),
                    entry.operating_system_version_imp,
                    entry.operating_system_version_types
                );
            }
            if (entry.is_at_least_imp != nullptr && entry.is_at_least_types != nullptr) {
                class_replaceMethod(
                    entry.cls,
                    @selector(isOperatingSystemAtLeastVersion:),
                    entry.is_at_least_imp,
                    entry.is_at_least_types
                );
            }
            if (
                entry.operating_system_version_string_imp != nullptr
                && entry.operating_system_version_string_types != nullptr
            ) {
                class_replaceMethod(
                    entry.cls,
                    @selector(operatingSystemVersionString),
                    entry.operating_system_version_string_imp,
                    entry.operating_system_version_string_types
                );
            }
        }

        bridge.installed = false;
        bridge.classes.clear();
        return 1;
    }
}

extern "C" int mts_metal_device_available() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        return device != nil ? 1 : 0;
    }
}

extern "C" int mts_mps_device_available() {
    @autoreleasepool {
        id<MTLDevice> device = MTLCreateSystemDefaultDevice();
        if (device == nil) {
            return 0;
        }
        return MPSSupportsMTLDevice(device) ? 1 : 0;
    }
}
