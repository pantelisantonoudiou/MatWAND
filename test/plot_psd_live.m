function plot_psd_live(pwr_mat,Flow,Fhigh,Fs)
% plot_psd_live(pwr_mat,Flow,Fhigh,Fs)
% plot_psd_live(proc_matrix,2,20,4000)

hold_samples = 10;

% get freq parameters
Flow = Flow*(5); Flow = ceil(Flow)+1;
Fhigh = Fhigh*(5); Fhigh = ceil(Fhigh)+1;
freq = 0:1/5:Fs/2;freqx = freq(Flow:Fhigh);

[~, nperiods] = size(pwr_mat);

cntr = 0;hold on

for i = 1:nperiods
      
        h1(i) = plot(freqx,pwr_mat(Flow:Fhigh,i),'k');
        xlabel('Freq. (Hz)') ;  ylabel ('Power (V^2 Hz^{-1})')
        title (i)
        axhand = gca;
        set(gcf,'color','w');
        axhand.Box = 'off';
        axhand.TickDir = 'out';
        axhand.XColor = 'k';
        axhand.YColor = 'k';
        axhand.FontName = 'Arial';
        axhand.FontSize = 14;
        axhand.FontWeight = 'bold';
        axhand.LineWidth = 1;
        
    if cntr > hold_samples
        set(h1(i-hold_samples),'Visible','off')      
    end    
    
    %refresh plot
%     refreshdata
    drawnow
    
    cntr = cntr+1;
    
end

end