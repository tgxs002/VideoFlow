## Refer to https://github.com/XiaoyuShi97/VideoFlow for basic usages

## inference optical flow, and export to npy
```
python inference.py --instruction_file <instruction_file> --input_folder ... --output_folder ... --batch_size 30 --padding_frames 4 --target_resolution 128
```
其中batch size是为了跑长视频用的，拆分video，逐个算，避免oom。padding frames必须是偶数，理论上越大越好，因为视频中间帧的flow性能比两边好。target resolution是最后输出光流的resolution，如果是-1就按照原resolution弄。
instruction_file格式如下：
```
[
    {
        "input_video": "1044_squeeze.mp4",
        "output_file": "1044_flow_out.npy"
    }
]
```

