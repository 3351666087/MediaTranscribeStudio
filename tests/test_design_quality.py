from __future__ import annotations

import json
from pathlib import Path
import shutil

from tools.design_quality.validator import (
    CANONICAL_EASE_DRAWER,
    CANONICAL_EASE_IN_OUT,
    CANONICAL_EASE_OUT,
    CANONICAL_EASINGS,
    COMPANION_SHA256,
    EXPECTED_LOCALES,
    REQUIRED_GLASS_TOKENS,
    REQUIRED_PALETTE_TOKENS,
    _background_findings,
    _desktop_companion_findings,
    _discover_imported_css,
    _glass_findings,
    _locale_findings,
    _motion_findings,
    _overview_structure_findings,
    audit_project,
)


ROOT = Path(__file__).resolve().parents[1]


def _codes(css: str, tmp_path: Path) -> set[str]:
    css_path = tmp_path / "fixture.css"
    css_path.write_text(css, encoding="utf-8")
    return {
        finding.code
        for finding in _motion_findings(tmp_path, css_path, css)
    }


def _write(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


def _background_fixture(tmp_path: Path) -> tuple[Path, str, Path, str, Path, str]:
    assets = tmp_path / "apps/desktop/src/assets"
    assets.mkdir(parents=True, exist_ok=True)
    for name in ("day.jpg", "night.jpg", "scene.png"):
        (assets / name).write_bytes(b"approved-background-fixture")

    app_path = _write(
        tmp_path / "apps/desktop/src/App.tsx",
        """
        import { SceneBackdrop } from "./components/SceneBackdrop";
        export function App() {
          return <>
            <SceneBackdrop />
            <SceneBackdrop />
            <SceneBackdrop />
          </>;
        }
        """,
    )
    component_path = _write(
        tmp_path / "apps/desktop/src/components/SceneBackdrop.tsx",
        """
        import dayArtwork from "../assets/day.jpg";
        import nightArtwork from "../assets/night.jpg";
        import sceneArtwork from "../assets/scene.png";
        export function SceneBackdrop() {
          return <div
            className="scene-backdrop"
            aria-hidden="true"
            data-background-contract="remote-day-night-scene"
          >
            <img
              className="scene-backdrop__theme scene-backdrop__theme--light"
              src={dayArtwork}
              alt=""
            />
            <img
              className="scene-backdrop__theme scene-backdrop__theme--dark"
              src={nightArtwork}
              alt=""
            />
            <img
              className="scene-backdrop__illustration"
              src={sceneArtwork}
              alt=""
            />
          </div>;
        }
        """,
    )
    css_path = _write(
        tmp_path / "apps/desktop/src/components/SceneBackdrop.css",
        """
        .scene-backdrop {
          position: fixed;
          inset: 0;
          overflow: hidden;
        }
        .scene-backdrop__theme,
        .scene-backdrop__illustration {
          object-fit: cover;
        }
        .scene-backdrop__theme {
          opacity: 0;
        }
        :root[data-theme="light"] .scene-backdrop__theme--light {
          opacity: 1;
        }
        :root[data-theme="dark"] .scene-backdrop__theme--dark {
          opacity: 1;
        }
        .scene-backdrop__illustration {
          opacity: 0.3;
        }
        @media (prefers-reduced-motion: reduce) {
          .scene-backdrop__theme,
          .scene-backdrop__illustration {
            transform: none;
          }
        }
        @media (forced-colors: active) {
          .scene-backdrop {
            display: none;
          }
        }
        """,
    )
    return (
        app_path,
        app_path.read_text(encoding="utf-8"),
        component_path,
        component_path.read_text(encoding="utf-8"),
        css_path,
        css_path.read_text(encoding="utf-8"),
    )


def _companion_fixture(tmp_path: Path) -> tuple[Path, str, Path, str, Path, str]:
    asset = tmp_path / "apps/desktop/src/assets/companion.gif"
    asset.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(
        ROOT / "apps/desktop/src/assets/companion.gif",
        asset,
    )
    app_path = _write(
        tmp_path / "apps/desktop/src/App.tsx",
        """
        import { DesktopCompanion } from "./components/DesktopCompanion";
        export function App() {
          return <DesktopCompanion
            collapseLabel={t("companion.collapse")}
            expandLabel={t("companion.expand")}
            imageAlt={t("companion.imageAlt")}
            regionLabel={t("companion.regionLabel")}
            statusText={t("companion.statusReady")}
          />;
        }
        """,
    )
    component_path = _write(
        tmp_path / "apps/desktop/src/components/DesktopCompanion.tsx",
        """
        import "./DesktopCompanion.css";
        const companionSource =
          new URL("../assets/companion.gif", import.meta.url).href;
        export interface DesktopCompanionProps {
          collapseLabel: string;
          expandLabel: string;
          imageAlt: string;
          regionLabel: string;
          statusText: string;
        }
        export function DesktopCompanion(props: DesktopCompanionProps) {
          const {
            collapseLabel,
            expandLabel,
            imageAlt,
            regionLabel,
            statusText,
          } = props;
          const collapsed = false;
          event.currentTarget.setPointerCapture(event.pointerId);
          event.currentTarget.releasePointerCapture(event.pointerId);
          return <aside aria-label={regionLabel}>
            <span role="status" aria-live="polite">{statusText}</span>
            <button
              type="button"
              aria-expanded={!collapsed}
              aria-keyshortcuts="ArrowUp ArrowDown ArrowLeft ArrowRight Enter Space"
              aria-label={collapsed ? expandLabel : collapseLabel}
            >
              <img src={companionSource} alt={imageAlt} />
            </button>
          </aside>;
        }
        """,
    )
    css_path = _write(
        tmp_path / "apps/desktop/src/components/DesktopCompanion.css",
        """
        @media (prefers-reduced-motion: reduce) {
          .desktop-companion,
          .desktop-companion__button {
            animation: none !important;
            transition: none !important;
          }
        }
        @media (forced-colors: active) {
          .desktop-companion__status,
          .desktop-companion__visual {
            color: CanvasText;
            background: Canvas;
            forced-color-adjust: auto;
          }
        }
        """,
    )
    return (
        app_path,
        app_path.read_text(encoding="utf-8"),
        component_path,
        component_path.read_text(encoding="utf-8"),
        css_path,
        css_path.read_text(encoding="utf-8"),
    )


def _locale_fixture() -> tuple[str, str]:
    locale_variables = {
        "en": "ENGLISH_MESSAGES",
        "zh-Hans": "zhHans",
        "zh-Hant": "zhHant",
        "ja": "ja",
        "ko": "ko",
        "es": "es",
        "fr": "fr",
        "de": "de",
        "pt-BR": "ptBr",
    }
    catalogs = []
    for index, (locale, variable) in enumerate(locale_variables.items()):
        if locale == "en":
            continue
        catalogs.append(
            f"""
            const {variable} = {{
              greeting: "{locale} hello {{name}}",
              progress: "{locale} {{done}} / {{total}}",
              repeated: "{locale} {{count}} then {{count}}",
            }};
            """
        )
    mappings = "\n".join(
        f'  "{locale}": {variable},'
        for locale, variable in locale_variables.items()
    )
    source = f"""
    export const SUPPORTED_LOCALES = [
      "en", "zh-Hans", "zh-Hant", "ja", "ko", "es", "fr", "de", "pt-BR",
    ] as const;
    export const ENGLISH_MESSAGES = {{
      greeting: "Hello {{name}}",
      progress: "{{done}} / {{total}}",
      repeated: "{{count}} then {{count}}",
    }};
    {"".join(catalogs)}
    type MessageKey = keyof typeof ENGLISH_MESSAGES;
    type MessageCatalog = Record<MessageKey, string>;
    export const MESSAGE_CATALOGS = {{
    {mappings}
    }};
    """
    core = """
    export function translate(locale, key) {
      const template = MESSAGE_CATALOGS[locale][key];
      if (template === undefined) {
        throw new Error(`Missing message ${locale}.${key}`);
      }
      return template;
    }
    """
    return source, core


def test_motion_policy_accepts_fast_explicit_pointer_gated_motion(
    tmp_path: Path,
) -> None:
    css = """
    .button {
      transition:
        transform 160ms ease-out,
        opacity 180ms ease-out,
        color 150ms ease;
    }
    @media (hover: hover) and (pointer: fine) {
      .button:hover { transform: translateY(-1px); }
    }
    .button:active { transform: scale(0.97); }
    @media (prefers-reduced-motion: reduce) {
      .button { transform: none !important; }
    }
    """
    assert _codes(css, tmp_path) == set()


def test_motion_policy_rejects_long_or_layout_affecting_motion(
    tmp_path: Path,
) -> None:
    css = """
    .bad {
      transition:
        height 200ms ease-out,
        transform 301ms ease-out,
        all 120ms ease-out;
    }
    """
    codes = _codes(css, tmp_path)
    assert "DQ-MOTION-PROPERTY" in codes
    assert "DQ-MOTION-DURATION" in codes


def test_hover_movement_requires_fine_pointer_media_query(tmp_path: Path) -> None:
    css = """
    .card { transition: transform 160ms ease-out; }
    .card:hover { transform: translateY(-2px); }
    @media (prefers-reduced-motion: reduce) {
      .card { transform: none !important; }
    }
    """
    assert "DQ-HOVER-MOTION-GATE" in _codes(css, tmp_path)


def test_reduced_motion_must_remove_transform_keyframe_movement(
    tmp_path: Path,
) -> None:
    css = """
    .toast { animation: toast-in 260ms ease-out both; }
    @keyframes toast-in {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (prefers-reduced-motion: reduce) {
      .toast { animation-duration: 0.01ms !important; }
    }
    """
    assert "DQ-REDUCED-MOTION-MOVEMENT" in _codes(css, tmp_path)


def test_reduced_motion_accepts_explicit_animation_removal(tmp_path: Path) -> None:
    css = """
    .toast { animation: toast-in 260ms ease-out both; }
    @keyframes toast-in {
      from { opacity: 0; transform: translateY(12px); }
      to { opacity: 1; transform: translateY(0); }
    }
    @media (prefers-reduced-motion: reduce) {
      .toast { animation: none !important; transform: none !important; }
    }
    """
    assert _codes(css, tmp_path) == set()


def test_ambient_loop_may_exceed_ui_limit_when_reduced_motion_disables_it(
    tmp_path: Path,
) -> None:
    css = """
    .ambient {
      animation: ambient-drift 18s ease-in-out infinite alternate;
    }
    @keyframes ambient-drift {
      from { opacity: 0.7; transform: translateX(-1px); }
      to { opacity: 1; transform: translateX(1px); }
    }
    @media (prefers-reduced-motion: reduce) {
      .ambient {
        animation: none !important;
        transform: none !important;
      }
    }
    """
    assert _codes(css, tmp_path) == set()


def test_ambient_loop_without_reduced_motion_is_rejected(tmp_path: Path) -> None:
    css = """
    .ambient {
      animation: ambient-drift 18s ease-in-out infinite alternate;
    }
    @keyframes ambient-drift {
      from { opacity: 0.7; transform: translateX(-1px); }
      to { opacity: 1; transform: translateX(1px); }
    }
    """
    codes = _codes(css, tmp_path)
    assert "DQ-MOTION-DURATION" not in codes
    assert "DQ-REDUCED-MOTION-MOVEMENT" in codes


def test_ambient_longhands_are_checked_like_animation_shorthand(
    tmp_path: Path,
) -> None:
    css = """
    .ambient {
      animation-name: ambient-drift;
      animation-duration: 20s;
      animation-iteration-count: infinite;
    }
    @keyframes ambient-drift {
      from { opacity: 0.6; }
      to { opacity: 1; }
    }
    @media (prefers-reduced-motion: reduce) {
      .ambient { animation: none !important; }
    }
    """
    assert _codes(css, tmp_path) == set()


def test_smooth_scroll_is_a_hard_failure(tmp_path: Path) -> None:
    assert "DQ-NAV-SMOOTH-SCROLL" in _codes(
        ".workspace-main { scroll-behavior: smooth; }",
        tmp_path,
    )


def test_imported_css_discovery_covers_global_and_component_styles(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "apps/desktop/src"
    _write(
        source_root / "main.tsx",
        """
        import "./styles/tokens.css";
        import "./styles/global.css";
        """,
    )
    _write(
        source_root / "components/SceneBackdrop.tsx",
        'import "./SceneBackdrop.css";',
    )
    _write(
        source_root / "components/DesktopCompanion.tsx",
        'import "./DesktopCompanion.css";',
    )
    for path in (
        source_root / "styles/tokens.css",
        source_root / "styles/global.css",
        source_root / "components/SceneBackdrop.css",
        source_root / "components/DesktopCompanion.css",
    ):
        _write(path, ":root { color: CanvasText; }")

    sources, findings = _discover_imported_css(tmp_path)
    assert findings == ()
    assert {
        source.path.relative_to(tmp_path).as_posix()
        for source in sources
    } == {
        "apps/desktop/src/styles/tokens.css",
        "apps/desktop/src/styles/global.css",
        "apps/desktop/src/components/SceneBackdrop.css",
        "apps/desktop/src/components/DesktopCompanion.css",
    }


def test_imported_css_discovery_fails_closed_on_missing_or_escaping_imports(
    tmp_path: Path,
) -> None:
    source_root = tmp_path / "apps/desktop/src"
    _write(
        source_root / "components/Broken.tsx",
        """
        import "./missing.css";
        import "../../../../outside.css";
        """,
    )

    _sources, findings = _discover_imported_css(tmp_path)
    codes = {finding.code for finding in findings}
    assert "DQ-CSS-IMPORT-MISSING" in codes
    assert "DQ-CSS-IMPORT-BOUNDARY" in codes
    assert "DQ-CSS-REQUIRED-IMPORT" in codes


def test_background_contract_requires_explicit_three_layer_visibility(
    tmp_path: Path,
) -> None:
    fixture = _background_fixture(tmp_path)
    assert _background_findings(tmp_path, *fixture) == []

    broken_css = fixture[-1].replace(
        ':root[data-theme="dark"] .scene-backdrop__theme--dark {\n'
        "          opacity: 1;",
        ':root[data-theme="dark"] .scene-backdrop__theme--dark {\n'
        "          opacity: 0;",
    )
    findings = _background_findings(tmp_path, *fixture[:-1], broken_css)
    assert "DQ-BACKGROUND-THEME-VISIBILITY" in {
        finding.code for finding in findings
    }


def test_desktop_companion_contract_checks_hash_accessibility_and_app_wiring(
    tmp_path: Path,
) -> None:
    fixture = _companion_fixture(tmp_path)
    assert _desktop_companion_findings(tmp_path, *fixture) == []
    assert (
        COMPANION_SHA256
        == "84555A0B2AD4B96C0282C50B5A3FD92D6AB3933E0FA72935E3FD59322389C4EB"
    )

    app_path, _app, component_path, component, css_path, css = fixture
    findings = _desktop_companion_findings(
        tmp_path,
        app_path,
        "export function App() { return null; }",
        component_path,
        component.replace('role="status"', 'role="note"'),
        css_path,
        css,
    )
    codes = {finding.code for finding in findings}
    assert "DQ-COMPANION-ACCESSIBILITY" in codes
    assert "DQ-COMPANION-APP-INTEGRATION" in codes


def test_complete_nine_locale_catalog_has_exact_key_and_placeholder_parity(
    tmp_path: Path,
) -> None:
    source, core = _locale_fixture()
    catalog_path = _write(tmp_path / "catalog.ts", source)
    core_path = _write(tmp_path / "core.ts", core)
    assert _locale_findings(
        tmp_path,
        catalog_path,
        source,
        core_path,
        core,
    ) == []


def test_locale_gate_resolves_relative_named_catalog_imports(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "completion.ts",
        """
        export const ZH_COMPLETION = {
          greeting: "你好 {name}",
          progress: "完成 {done} / {total}",
          repeated: "{count} 然后 {count}",
        } as const;
        """,
    )
    source, core = _locale_fixture()
    source = (
        'import { ZH_COMPLETION } from "./completion";\n'
        + source.replace(
            """const zhHans = {
              greeting: "zh-Hans hello {name}",
              progress: "zh-Hans {done} / {total}",
              repeated: "zh-Hans {count} then {count}",
            };""",
            "const zhHans = { ...ZH_COMPLETION };",
        )
    )
    catalog_path = _write(tmp_path / "catalog.ts", source)
    core_path = _write(tmp_path / "core.ts", core)

    assert _locale_findings(
        tmp_path,
        catalog_path,
        source,
        core_path,
        core,
    ) == []


def test_locale_gate_resolves_auditable_json_catalog_fragments(
    tmp_path: Path,
) -> None:
    locales = (
        "en",
        "zh-Hans",
        "zh-Hant",
        "ja",
        "ko",
        "es",
        "fr",
        "de",
        "pt-BR",
    )
    _write(
        tmp_path / "feature.json",
        json.dumps(
            {
                locale: {
                    "feature.title": (
                        "Feature {count}"
                        if locale == "en"
                        else f"{locale} feature {{count}}"
                    )
                }
                for locale in locales
            },
            ensure_ascii=False,
        ),
    )
    source, core = _locale_fixture()
    source = 'import featureMessages from "./feature.json";\n' + source
    source = source.replace(
        'repeated: "{count} then {count}",',
        'repeated: "{count} then {count}",\n'
        "      ...featureMessages.en,",
    )
    locale_references = {
        "zh-Hans": 'featureMessages["zh-Hans"]',
        "zh-Hant": 'featureMessages["zh-Hant"]',
        "ja": "featureMessages.ja",
        "ko": "featureMessages.ko",
        "es": "featureMessages.es",
        "fr": "featureMessages.fr",
        "de": "featureMessages.de",
        "pt-BR": 'featureMessages["pt-BR"]',
    }
    for locale, reference in locale_references.items():
        source = source.replace(
            f'repeated: "{locale} {{count}} then {{count}}",',
            f'repeated: "{locale} {{count}} then {{count}}",\n'
            f"              ...{reference},",
        )

    catalog_path = _write(tmp_path / "catalog.ts", source)
    core_path = _write(tmp_path / "core.ts", core)

    assert _locale_findings(
        tmp_path,
        catalog_path,
        source,
        core_path,
        core,
    ) == []


def test_locale_gate_rejects_missing_keys_and_placeholder_multiplicity(
    tmp_path: Path,
) -> None:
    source, core = _locale_fixture()
    source = source.replace(
        'repeated: "zh-Hans {count} then {count}",',
        'repeated: "zh-Hans {count}",',
    ).replace(
        'progress: "fr {done} / {total}",',
        "",
    )
    catalog_path = _write(tmp_path / "catalog.ts", source)
    core_path = _write(tmp_path / "core.ts", core)
    findings = _locale_findings(
        tmp_path,
        catalog_path,
        source,
        core_path,
        core,
    )
    codes = {finding.code for finding in findings}
    assert "DQ-I18N-KEY-MISSING" in codes
    assert "DQ-I18N-PLACEHOLDER" in codes


def test_locale_gate_rejects_partial_catalogs_and_english_fallback(
    tmp_path: Path,
) -> None:
    source, _core = _locale_fixture()
    source = source.replace(
        "type MessageCatalog = Record<MessageKey, string>;",
        "type MessageCatalog = Partial<Record<MessageKey, string>>;",
    )
    core = """
    const english = ENGLISH_MESSAGES[key];
    const localized = MESSAGE_CATALOGS[locale][key];
    const template = localized ?? english;
    """
    catalog_path = _write(tmp_path / "catalog.ts", source)
    core_path = _write(tmp_path / "core.ts", core)
    findings = _locale_findings(
        tmp_path,
        catalog_path,
        source,
        core_path,
        core,
    )
    codes = {finding.code for finding in findings}
    assert "DQ-I18N-PARTIAL-CATALOG" in codes
    assert "DQ-I18N-ENGLISH-FALLBACK" in codes


def test_locale_gate_rejects_non_english_catalog_spread_fallback(
    tmp_path: Path,
) -> None:
    source, core = _locale_fixture()
    source = source.replace(
        """const zhHans = {
              greeting: "zh-Hans hello {name}",
              progress: "zh-Hans {done} / {total}",
              repeated: "zh-Hans {count} then {count}",
            };""",
        "const zhHans = { ...ENGLISH_MESSAGES };",
    )
    catalog_path = _write(tmp_path / "catalog.ts", source)
    core_path = _write(tmp_path / "core.ts", core)
    findings = _locale_findings(
        tmp_path,
        catalog_path,
        source,
        core_path,
        core,
    )
    assert "DQ-I18N-ENGLISH-SPREAD-FALLBACK" in {
        finding.code for finding in findings
    }


def test_overview_structure_accepts_shared_portal_workspace(
    tmp_path: Path,
) -> None:
    overview_path = _write(
        tmp_path / "OverviewWorkspace.tsx",
        """
        <div className="studio-room-grid studio-room-portals">
          <button className={`studio-room-portal studio-room-portal--${tone}`}>
            <span className="studio-room-portal__track" />
            <span className="studio-room-portal__footer" />
          </button>
        </div>
        """,
    )
    css_path = _write(
        tmp_path / "global.css",
        """
        .studio-room-grid {
          gap: 0;
          border: 1px solid;
          background: rgba(255, 255, 255, 0.5);
          backdrop-filter: blur(20px);
          overflow: clip;
        }
        .studio-room-portal {
          border-right: 1px solid;
          border-radius: 0;
          background: transparent;
          box-shadow: none;
        }
        .studio-room-portal__track {
          position: absolute;
          inset: 0;
          pointer-events: none;
        }
        """,
    )
    overview = overview_path.read_text(encoding="utf-8")
    css = css_path.read_text(encoding="utf-8")
    assert _overview_structure_findings(
        tmp_path,
        overview_path,
        overview,
        css_path,
        css,
    ) == []

    findings = _overview_structure_findings(
        tmp_path,
        overview_path,
        overview + '<article className="studio-room-card" />',
        css_path,
        css,
    )
    assert "DQ-STRUCTURE-CARD-WALL" in {
        finding.code for finding in findings
    }


def test_live_repository_audit_is_fail_closed_and_does_not_claim_pack_passage() -> None:
    report = audit_project(ROOT)
    payload = report.as_dict()

    assert payload["policy"]["failClosed"] is True
    assert payload["policy"]["maximumUiMotionMs"] == 300
    assert payload["externalDesignPack"]["passageClaimed"] is False
    assert payload["externalDesignPack"]["status"] == "blocked-upstream-v3-contract"
    assert payload["externalDesignPack"]["blocker"] == (
        "repairCatalog.passPolicy is required"
    )
    assert {check.id for check in report.checks} == {
        "css-source-graph",
        "motion",
        "frequent-navigation",
        "forced-colors",
        "background-assets",
        "desktop-companion",
        "glass-and-palette",
        "page-hierarchy",
        "overview-structure",
        "locales",
        "native-screenshot-evidence",
        "background-ocr-evidence",
    }
    assert not any(
        finding.code == "DQ-INTERNAL-FAIL-CLOSED"
        for finding in report.findings
    )


def test_release_mode_requires_real_screenshot_and_ocr_manifests() -> None:
    report = audit_project(ROOT, require_evidence=True)
    codes = {finding.code for finding in report.findings}
    assert "DQ-SCREENSHOT-EVIDENCE-MISSING" in codes
    assert "DQ-OCR-EVIDENCE-MISSING" in codes
    assert report.status == "fail"


def test_live_repository_locale_audit_is_policy_driven_not_internal_failure() -> None:
    report = audit_project(ROOT)
    locale_check = next(check for check in report.checks if check.id == "locales")

    assert all(
        finding.code != "DQ-INTERNAL-FAIL-CLOSED"
        for finding in locale_check.findings
    )


def test_all_canonical_easing_curves_are_required_design_pack_curves(
    tmp_path: Path,
) -> None:
    assert CANONICAL_EASE_OUT == "cubic-bezier(0.23, 1, 0.32, 1)"
    assert CANONICAL_EASE_IN_OUT == "cubic-bezier(0.77, 0, 0.175, 1)"
    assert CANONICAL_EASE_DRAWER == "cubic-bezier(0.32, 0.72, 0, 1)"

    tokens = "\n".join(
        (
            ":root {",
            *(
                f"  {token}: "
                f"{'linear' if token in CANONICAL_EASINGS else '#fff'};"
                for token in sorted(
                    REQUIRED_GLASS_TOKENS
                    | REQUIRED_PALETTE_TOKENS
                    | set(CANONICAL_EASINGS)
                )
            ),
            "}",
            ':root[data-theme="dark"] {',
            *(
                f"  {token}: #000;"
                for token in sorted(
                    REQUIRED_GLASS_TOKENS | REQUIRED_PALETTE_TOKENS
                )
            ),
            "}",
        )
    )
    token_path = _write(tmp_path / "tokens.css", tokens)
    css_path = _write(tmp_path / "global.css", ":root { color: black; }")
    findings = _glass_findings(
        tmp_path,
        token_path,
        tokens,
        css_path,
        css_path.read_text(encoding="utf-8"),
    )
    assert {
        finding.evidence.get("token")
        for finding in findings
        if finding.code == "DQ-EASING-TOKEN"
    } == set(CANONICAL_EASINGS)
