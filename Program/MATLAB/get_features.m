function features = get_features(signal,Ts)
% @function get_features(signal)
% @brief get time/frequency domain features
% @parameter signal: 1D array type(column array recommended) signal
% @parameter Ts: Sampling rate/ Sampling time
% @return struct type features
features.dimension = size(signal);
if features.dimension(2)>1 && features.dimension(1)>1
    disp("Size of signal is ambiguous")
    signal = signal(:,1);
end
N =features.dimension(1);
features.mean      = mean(signal);
features.std       = std(signal);
features.rms       = sqrt(sum(signal.^2)/N);
features.sra       = (sum(sqrt(abs(signal)))/N).^2;
features.aav       = sum(abs(signal))/N;
features.energy    = sum(signal.^2);
features.peak      = max(signal);
features.ppv       = peak2peak(signal);
features.if        = features.peak/features.aav;
features.sf        = features.rms /features.aav;
features.cf        = features.peak/features.rms;
features.mf        = features.peak/features.sra;
features.sk        = skewness(signal);
features.kt        = kurtosis(signal);
% FFT
p1 = fft(signal);
p1(2:end-1)= 2.0 * p1(2:end-1);
features.fft = p1;
Fs = 1/Ts;
frequency_max=Fs/2; 
frequency_resolution=Fs/N; 
features.frequency = 0:frequency_resolution:frequency_max;
figure("FFT"); hold on; grid on;
plot(features.frequency, features.fft);
xlabel("Frequency [Hz]",'FontWeight','bold');
ylabel("Amplitude [-]",'FontWeight','bold');
features.rmsf= sqrt(1/N *sum(xfeature.fft.^2));
% Hilbert Transform
x = signal(:); %serialize
N = length(x);
X = zeros(N,1);
P= fft(x);     
X(1) = P(1);
X(2:N/2) = 2*P((2:N/2));
X(N/2+1) = P(N/2+1);
features.Hilbert.analytic_signal   = ifft(X,N);
features.Hilbert.instant_amplitude = abs(z); 

end

