ffmpeg -framerate 30 -i damped_wave/openmp/sim/frame_%05d.pgm -c:v libx264 -pix_fmt yuv420p nome_video.mp4
