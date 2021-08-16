
modify_setting(){


  original_file=$1
  modified_file_folder=$2
  modified_file=$modified_file_folder/$3
  if [[ ! -d $modified_file_folder/ ]]; then
    mkdir -p $modified_file_folder/
  fi

  line_number_of_sampling_number=106
  sampling_number=1000
  line_number_of_sampling_seed=100
  rand_time=$4
  line_number_of_outputfolder=127
  output_folder=$5


  echo cp $original_file $modified_file
  cp $original_file $modified_file
  # sed $line_number_of_sampling_number's/.*/Sampling.number_of_sampling   : '$sampling_number'/' $modified_file  > $modified_file.tmp && mv $modified_file.tmp $modified_file
  sed $line_number_of_sampling_seed's/.*/Sampling.random_seed          : '$rand_time'/' $modified_file  > $modified_file.tmp 
  mv $modified_file.tmp $modified_file
  sed $line_number_of_outputfolder's/.*/OutputFolder                    : '${output_folder//\//\\/}'/'             $modified_file  > $modified_file.tmp
  mv $modified_file.tmp $modified_file
}

modify_setting_if(){
  original_setting_file=$(pwd)/settings/TUM/TUM1.yaml 
  setting_file_folder=$1
  setting_file=$2
  rand_time=$3
  output_path=$4

  modify_setting $original_setting_file $setting_file_folder $setting_file $rand_time $output_path
}

run_tum_all_seq(){
  output_root_folder=$1
  rand_time=$2
  setting_file=$3

  # run_vo desk  $rand_time $output_root_folder $setting_file
  # run_vo desk2 $rand_time $output_root_folder $setting_file
  # run_vo rpy   $rand_time $output_root_folder $setting_file
  # run_vo xyz   $rand_time $output_root_folder $setting_file
  run_vo 360   $rand_time $output_root_folder $setting_file
  # run_vo plant $rand_time $output_root_folder $setting_file
  # run_vo room  $rand_time $output_root_folder $setting_file
  wait


}

evaluate_tum(){
  seq=$1
  py=python
  result_root=$2
  dataset_seq_root=$3
  estimation_file=$4
  result_sub_folder=${seq}
  tum_tool_path=$(pwd)/settings/TUM/evaluation_tool
  output_file=${result_root}/rpeate_summary_$5.txt
  gt_file=${dataset_seq_root}/groundtruth.txt
  if [[ ! -d ${result_root}/ ]]; then
    mkdir -p ${result_root}/
  fi
  cd ${tum_tool_path}/
  echo "run rpe ....." > ${output_file}
  ${py} evaluate_rpe.py --fixed_delta --delta_unit s --verbose --save ${estimation_file}_rpe_analysis.txt --plot ${estimation_file}.png  ${gt_file} ${estimation_file} >> ${output_file}
  echo "run ate ....." >> ${output_file}
  ${py} evaluate_ate.py --save_associations ${estimation_file}_ate_analysis.txt --plot ${estimation_file}_ate.png --verbose   ${gt_file} ${estimation_file} >> ${output_file}
  echo "run rpe ....." > ${output_file}_f
  ${py} evaluate_rpe.py --fixed_delta --delta 1 --delta_unit f --verbose --save ${estimation_file}_rpe_analysis_f.txt --plot ${estimation_file}_f.png  ${gt_file} ${estimation_file} >> ${output_file}_f

  # echo "run rpe ....." > ${output_file}
  # ${py} evaluate_rpe.py --fixed_delta --delta_unit s --verbose --save ${estimation_file}_rpe_analysis.txt --plot ${estimation_file}.png  ${gt_file} /home/vlsilab/linux_files/projects/_ICPORB_VO/ORBTraj.txt >> ${output_file}
  # echo "run ate ....." >> ${output_file}
  # ${py} evaluate_ate.py --save_associations ${estimation_file}_ate_analysis.txt --plot ${estimation_file}_ate.png --verbose   ${gt_file}  /home/vlsilab/linux_files/projects/_ICPORB_VO/ORBTraj.txt>> ${output_file}
  #
  # echo "run rpe ....." > ${output_file}_f
  # ${py} evaluate_rpe.py --fixed_delta --delta 1 --delta_unit f --verbose --save ${estimation_file}_rpe_analysis_f.txt --plot ${estimation_file}_f.png  ${gt_file} /home/vlsilab/linux_files/projects/_ICPORB_VO/ORBTraj.txt >> ${output_file}_f
  #
  cat ${output_file}
  cd -
}

run_vo(){

  seq=$1
  rand_time=$2
  output_file_root=$3/${seq}
  associations_file=$(pwd)/settings/TUM/associations/rs_rgb_${seq}_association.txt
  dataset_seq_root=$(pwd)/../../dataset/${seq}
  setting_file=$4
  output_file=${output_file_root}/${seq}_traj${rand_time}.txt
  if [[ ! -d ${output_file_root} ]]; then
    mkdir -p ${output_file_root}
  fi

  build/ICPORBVO_example ${associations_file} ${dataset_seq_root} ${setting_file} ${output_file} ./Thirdparty/ORB_SLAM2/Vocabulary/ORBvoc.txt
  #build/ICPVO_example ${associations_file} ${dataset_seq_root} ${setting_file} ${output_file} 
  # echo evaluate_tum ${seq} ${output_file_root} ${dataset_seq_root} ${output_file} ${rand_time}
  evaluate_tum ${seq} ${output_file_root} ${dataset_seq_root} ${output_file} ${rand_time}

}

run_all_config(){
  loop_cnt=$1
  config_folder_name=eas_$1
  config_folder=EASTUM_ICPORBFINAL
  output_root_folder=$(pwd)/vo_results/$config_folder/$config_folder_name
  setting_file_folder=$(pwd)/settings/TUM/tum_settings
  for (( i = 1; i < loop_cnt+1; i++ )); do
    setting_file1=TUM_$config_folder_name\_$i.yaml
    setting_file_comb=$setting_file_folder/$setting_file1
    # echo $setting_file1
    modify_setting_if $setting_file_folder $setting_file1 $i $output_root_folder
    run_tum_all_seq $output_root_folder $i $setting_file_comb
  done
  wait
}

run_all_config 1
