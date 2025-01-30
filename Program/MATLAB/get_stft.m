function [S,time,frequency]=get_stft(signal,Fs)
% @Author Jin Kwak
% @E-mail 104green@naver.com || superbjin000105@gmail.com
% @brief show spectogram and calculate short time fourier transform of signal
% @parameter signal: signal of 1D array
% @parameter Fs: Sampling Frequency
% @return S: short time fourier transform result
% @return time: time
% @return frequency: frequency of the spectrum
N= length(signal);
Ts = 1/Fs;
[S, time, frequency] = stft(signal,Fs);
figure("STFT"); hold on; grid on;
waterfall(frequency,time,abs(S)');

end

