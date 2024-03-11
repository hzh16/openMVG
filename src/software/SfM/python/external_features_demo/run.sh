data_dir=/home/song/Documents/zihang/20240305/ImageDataset_SceauxCastle/images
out_dir=/home/song/Documents/zihang/20240311/output_deeplabv3
camera_database=/home/song/Documents/zihang/20240305/openMVG/src/openMVG/exif/sensor_width_database/sensor_width_camera_database.txt
# Can be either DeepLabv3 or DINOv2
descriptor_type=DeepLabv3

recon_dir=$out_dir/reconstruction_sequential
mkdir $out_dir
mkdir $recon_dir

# 1. Intrinsics analysis
echo "1. Intrinsics analysis"
openMVG_main_SfMInit_ImageListing \
    -i "$data_dir" \
    -o "$out_dir" \
    -d "$camera_database"

# 2. Compute features
echo "2. Compute features"
python kornia_deep_features.py \
    -i "$out_dir/sfm_data.json" \
    -m "$out_dir" \
    --deep_descriptor_type $descriptor_type

# 3. Compute matching pairs
echo "3. Compute matching pairs"
openMVG_main_PairGenerator \
    -i "$out_dir/sfm_data.json" \
    -o "$out_dir/pairs.bin"

# 4. Compute matches
echo "4. Compute matches"
openMVG_main_ComputeMatches \
    -i "$out_dir/sfm_data.json" \
    -p "$out_dir/pairs.bin" \
    -o "$out_dir/matches.putative.bin" \
    -d $descriptor_type

# 5. Filter matches
echo "5. Filter matches"
openMVG_main_GeometricFilter \
    -i "$out_dir/sfm_data.json" \
    -m "$out_dir/matches.putative.bin" \
    -g f \
    -o "$out_dir/matches.f.bin" \
    -d $descriptor_type

# 6. Do Sequential/Incremental reconstruction
echo "6. Do Sequential/Incremental reconstruction"
openMVG_main_SfM \
    --sfm_engine INCREMENTAL \
    --input_file "$out_dir/sfm_data.json" \
    --match_dir "$out_dir" \
    --output_dir "$recon_dir"

# 7. Colorize Structure
echo "7. Colorize Structure"
openMVG_main_ComputeSfM_DataColor \
    -i "$recon_dir/sfm_data.bin" \
    -o "$recon_dir/colorized.ply"

# 8. Convert SfM scene from OpenMVG to OpenMVS
echo "8. Convert SfM scene from OpenMVG to OpenMVS"
openMVG_main_openMVG2openMVS \
    -i "$recon_dir/sfm_data.bin" \
    -o "$recon_dir/scene.mvs" \
    -d "$recon_dir/scene_undistorted_images"

# 9. Dense Point-Cloud Reconstruction
echo "9. Dense Point-Cloud Reconstruction"
# We must execute this command in the directory
old_dir=`pwd`
cd $recon_dir
/usr/local/bin/OpenMVS/DensifyPointCloud scene.mvs
cd $old_dir
