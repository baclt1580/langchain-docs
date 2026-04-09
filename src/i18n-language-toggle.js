/**
 * Lightweight language switcher for translated docs pages.
 *
 * URL rules:
 * - Default language: /path
 * - Translated language: /i18n/<language-code>/path
 */
(function () {
  "use strict";

  const DEFAULT_CONFIG = {
    defaultLanguage: "en",
    prefix: "i18n",
    languages: [
      { code: "en", label: "English" },
      { code: "zh-CN", label: "Chinese (Simplified)" },
    ],
  };
  const SWITCHER_ID = "docs-i18n-switcher";
  const PREFIX_DELIMITER = "/";

  let config = DEFAULT_CONFIG;
  let currentLanguage = DEFAULT_CONFIG.defaultLanguage;

  function isAbsoluteInternalPath(pathname) {
    return pathname.startsWith("/") && !pathname.startsWith("//");
  }

  function isSpecialPath(pathname) {
    const lower = pathname.toLowerCase();
    if (lower.startsWith("/images/") || lower.startsWith("/fonts/")) {
      return true;
    }

    return /\.(png|jpe?g|gif|svg|ico|css|js|json|xml|txt|woff2?|ttf)$/i.test(lower);
  }

  function normalizePath(pathname) {
    if (!pathname) {
      return "/";
    }

    if (!pathname.startsWith("/")) {
      return "/" + pathname;
    }

    return pathname;
  }

  function getAllowedLanguageCodes() {
    return new Set(config.languages.map((item) => item.code));
  }

  function stripLanguagePrefix(pathname) {
    const normalized = normalizePath(pathname);
    const prefix = `/${config.prefix}/`;
    if (!normalized.startsWith(prefix)) {
      return { path: normalized, language: config.defaultLanguage };
    }

    const rest = normalized.slice(prefix.length);
    const slashIndex = rest.indexOf(PREFIX_DELIMITER);
    const languageCode = slashIndex === -1 ? rest : rest.slice(0, slashIndex);

    if (!getAllowedLanguageCodes().has(languageCode)) {
      return { path: normalized, language: config.defaultLanguage };
    }

    const remainder = slashIndex === -1 ? "/" : `/${rest.slice(slashIndex + 1)}`;
    return { path: normalizePath(remainder), language: languageCode };
  }

  function localizePath(pathname, languageCode) {
    const normalized = normalizePath(pathname);
    const stripped = stripLanguagePrefix(normalized).path;

    if (languageCode === config.defaultLanguage) {
      return stripped;
    }

    if (stripped === "/") {
      return `/${config.prefix}/${languageCode}/`;
    }

    return `/${config.prefix}/${languageCode}${stripped}`;
  }

  function rewriteInternalHref(href, languageCode) {
    const url = new URL(href, window.location.origin);
    if (url.origin !== window.location.origin) {
      return null;
    }

    if (!isAbsoluteInternalPath(url.pathname) || isSpecialPath(url.pathname)) {
      return null;
    }

    const rewrittenPath = localizePath(url.pathname, languageCode);
    return `${rewrittenPath}${url.search}${url.hash}`;
  }

  function updateCurrentLanguageFromPath() {
    currentLanguage = stripLanguagePrefix(window.location.pathname).language;
    const select = document.getElementById(SWITCHER_ID);
    if (select) {
      select.value = currentLanguage;
    }
  }

  function createSwitcher() {
    if (document.getElementById(SWITCHER_ID)) {
      return;
    }

    const container = document.createElement("div");
    container.style.position = "fixed";
    container.style.right = "16px";
    container.style.bottom = "16px";
    container.style.zIndex = "9999";
    container.style.padding = "8px";
    container.style.borderRadius = "10px";
    container.style.background = "rgba(255, 255, 255, 0.95)";
    container.style.border = "1px solid rgba(0, 0, 0, 0.12)";
    container.style.boxShadow = "0 8px 24px rgba(0, 0, 0, 0.12)";

    const select = document.createElement("select");
    select.id = SWITCHER_ID;
    select.style.border = "none";
    select.style.background = "transparent";
    select.style.fontSize = "13px";
    select.style.cursor = "pointer";
    select.style.outline = "none";

    config.languages.forEach((language) => {
      const option = document.createElement("option");
      option.value = language.code;
      option.textContent = language.label;
      select.appendChild(option);
    });

    select.value = currentLanguage;
    select.addEventListener("change", (event) => {
      const target = event.target;
      if (!(target instanceof HTMLSelectElement)) {
        return;
      }

      const localizedPath = localizePath(window.location.pathname, target.value);
      const nextUrl = `${localizedPath}${window.location.search}${window.location.hash}`;
      window.location.assign(nextUrl);
    });

    container.appendChild(select);
    document.body.appendChild(container);
  }

  function installLinkInterceptor() {
    document.addEventListener(
      "click",
      (event) => {
        if (
          event.defaultPrevented ||
          event.button !== 0 ||
          event.metaKey ||
          event.ctrlKey ||
          event.shiftKey ||
          event.altKey
        ) {
          return;
        }

        const target = event.target;
        if (!(target instanceof Element)) {
          return;
        }

        const anchor = target.closest("a[href]");
        if (!(anchor instanceof HTMLAnchorElement)) {
          return;
        }

        const href = anchor.getAttribute("href");
        if (
          !href ||
          href.startsWith("#") ||
          href.startsWith("mailto:") ||
          href.startsWith("tel:")
        ) {
          return;
        }

        if (anchor.target && anchor.target !== "_self") {
          return;
        }

        const localized = rewriteInternalHref(href, currentLanguage);
        if (!localized || localized === href) {
          return;
        }

        event.preventDefault();
        window.location.assign(localized);
      },
      true,
    );
  }

  function installPathChangeListeners() {
    const originalPushState = history.pushState;
    const originalReplaceState = history.replaceState;

    const onPathChange = () => {
      updateCurrentLanguageFromPath();
    };

    history.pushState = function (...args) {
      originalPushState.apply(this, args);
      onPathChange();
    };

    history.replaceState = function (...args) {
      originalReplaceState.apply(this, args);
      onPathChange();
    };

    window.addEventListener("popstate", onPathChange);
  }

  async function loadConfig() {
    try {
      const response = await fetch("/i18n-config.json", { cache: "no-store" });
      if (!response.ok) {
        return;
      }

      const parsed = await response.json();
      if (
        !parsed ||
        typeof parsed !== "object" ||
        !Array.isArray(parsed.languages) ||
        parsed.languages.length === 0
      ) {
        return;
      }

      config = {
        defaultLanguage:
          typeof parsed.defaultLanguage === "string"
            ? parsed.defaultLanguage
            : DEFAULT_CONFIG.defaultLanguage,
        prefix:
          typeof parsed.prefix === "string" && parsed.prefix
            ? parsed.prefix
            : DEFAULT_CONFIG.prefix,
        languages: parsed.languages.filter(
          (language) =>
            language &&
            typeof language.code === "string" &&
            language.code &&
            typeof language.label === "string" &&
            language.label,
        ),
      };

      if (!config.languages.some((language) => language.code === config.defaultLanguage)) {
        config.languages.unshift({
          code: config.defaultLanguage,
          label: config.defaultLanguage,
        });
      }
    } catch (_error) {
      // Keep default config when i18n config is unavailable.
    }
  }

  async function init() {
    await loadConfig();
    updateCurrentLanguageFromPath();
    createSwitcher();
    installLinkInterceptor();
    installPathChangeListeners();
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", () => {
      void init();
    });
  } else {
    void init();
  }
})();
