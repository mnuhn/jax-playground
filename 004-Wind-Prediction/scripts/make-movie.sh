ffmpeg -framerate 30 -pattern_type glob -i 'png/0*.png' \
  -vf "pad=ceil(iw/2)*2:ceil(ih/2)*2" \
  -c:v libx264 -pix_fmt yuv420p out.mp4
