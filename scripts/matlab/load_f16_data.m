csv_filepath = '/Users/jack/f16-gvt/data/F16GVT_Files/BenchmarkData/F16Data_SineSw_Level1.csv';
data = readtable(csv_filepath);
max_k = 107005;
d = data.Force(1:max_k);
e = [data.Acceleration1(1:max_k), data.Acceleration2(1:max_k),data.Acceleration3(1:max_k)];
ts = 1/data.Fs(1);