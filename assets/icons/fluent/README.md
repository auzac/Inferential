# Interface icons

20 px *regular* icons from Microsoft's Fluent UI System Icons
(https://github.com/microsoft/fluentui-system-icons, MIT License, see `LICENSE`).

`src/inferential_studio/ui/icons.py` loads them by file name (without `.svg`) and recolours
the `#212121` fill to the theme's text colour. To add one, download
`assets/<Icon Name>/SVG/ic_fluent_<name>_20_regular.svg` from that repository and save it
here as `<name>.svg`.
