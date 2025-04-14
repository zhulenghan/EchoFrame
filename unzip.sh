cd /mnt/new_volume/vgg_sound/

for file in vggsound_*.tar.gz; do
    echo "🔄 解压 $file"
    tar -xzf "$file" --remove-files
done

