# Dev shell for the Tauri + Svelte research dashboard.
#
# Tauri's Rust backend links the system WebKit/GTK stack via pkg-config, none of which is available
# globally on NixOS — this shell provides it. The sim crates (neural-sim/neural-telemetry/neural-cli)
# build with a plain `cargo` and don't need any of this.
#
# Usage:
#   nix-shell                       # from the repo root
#   cd dashboard && npm install     # one-time JS deps
#   npm run tauri dev               # launch the dashboard
{ pkgs ? import <nixpkgs> { } }:
let
  # Runtime/link libraries the webview pulls in. Tauri 2 uses webkitgtk 4.1 + libsoup 3
  # (NOT the 4.0 / soup2 pair that older Tauri 1 guides reference).
  libs = with pkgs; [
    webkitgtk_4_1
    gtk3
    cairo
    gdk-pixbuf
    glib
    pango
    harfbuzz
    libsoup_3
    at-spi2-atk
    librsvg
    openssl
  ];
in
pkgs.mkShell {
  # pkg-config is the missing native tool every *-sys build script needs; cargo-tauri provides the
  # `cargo tauri` CLI; nodejs runs the Vite/Svelte frontend.
  nativeBuildInputs = with pkgs; [
    pkg-config
    gobject-introspection
    cargo-tauri
    nodejs_22
  ];
  buildInputs = libs;

  shellHook = ''
    # WebKit on many GPUs/drivers renders a blank window without this.
    export WEBKIT_DISABLE_COMPOSITING_MODE=1
    # Let the webview find gsettings schemas (otherwise GTK aborts on startup).
    export XDG_DATA_DIRS="${pkgs.gsettings-desktop-schemas}/share/gsettings-schemas/${pkgs.gsettings-desktop-schemas.name}:${pkgs.gtk3}/share/gsettings-schemas/${pkgs.gtk3.name}:$XDG_DATA_DIRS"
  '';

  # dlopen-ed at runtime by the webview.
  LD_LIBRARY_PATH = pkgs.lib.makeLibraryPath libs;
}
