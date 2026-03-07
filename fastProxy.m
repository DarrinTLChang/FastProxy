function [indexAll,par] = fastProxy(directoryRead,microLabels,par,directorySave,mode,considered_window,ASO_switch,fs)

    
    % Initialize arrays
    numChannels = length(microLabels);
    indexAll = cell(numChannels,1); 
    thresholdAll = cell(numChannels,1);
    thersoldArtifacthAll = cell(numChannels,1);
    fs_array =zeros(size(numChannels));
    biasIndex = considered_window.start-1;
        
    % start parallel pooling
%     delete(gcp('nocreate'));
%     parpool(par.parpoolworkers); 

    if isempty(gcp('nocreate'))
        parpool(par.parpoolworkers);
    end
    
    % Rearrange variables for parallel processing
    filterCutoffs = [par.detect_fmin par.detect_fmax]; % detection bandpass filter range
    filterOrder = par.detect_order; % detection bandpass filter order

    %% 1. Loop through micro channels and do threshold detection
    fprintf('Detecting events.\n')
    parfor i = 1:numChannels                                                         %% was parfor
        % load micro channel data
        queryfile=dir([directoryRead filesep microLabels{i} '*.mat']);

%         microFile = load([queryfile.folder filesep queryfile.name]);
        objFile = matfile([queryfile.folder filesep queryfile.name]);
        data = objFile.data(1,considered_window.start:considered_window.end);

        % Apply band-pass filter for detection
        [filtData,~] =  butterfilter (filterCutoffs,fs,filterOrder,data,'bandpass');

        % Apply Nonlinear energy operator
        % switch par.neo_order
        %     case 1
        %         neoDataRaw = sqrt(abs(filtData.^2-circshift(filtData,1).*circshift(filtData,-1)));
        %     case 2
        %         neoDataRaw = nthroot(abs(filtData.^4-circshift(filtData,1).*circshift(filtData,-1).* ...
        %         circshift(filtData,2).*circshift(filtData,-2)),4);
        %     case 3
        %         neoDataRaw = nthroot(abs(filtData.^6-circshift(filtData,1).*circshift(filtData,-1).* ...
        %         circshift(filtData,2).*circshift(filtData,-2).* ...
        %         circshift(filtData,3).*circshift(filtData,-3)),6); 
        %     otherwise
        %         fprintf('\nNEO order parameter is not supported. Using the default order (1)\n')
        %         neoDataRaw = sqrt(abs(filtData.^2-circshift(filtData,1).*circshift(filtData,-1)));
        % end
        if ASO_switch == 1
            asoDataRaw = filtData .* (filtData - circshift(filtData,1));
            neoDataRaw = asoDataRaw
        else
            neoDataRaw = sqrt(abs(filtData.^2-circshift(filtData,1).*circshift(filtData,-1)));
        end
 
        % Get spike indexes
        [indexAll{i},thresholdAll{i},thersoldArtifacthAll{i}] = getSpikeIndex(neoDataRaw,fs,par,mode,i); 
        fprintf([microLabels{i} ' -> %i events detected.\n'],numel(indexAll{i}))
        fs_array(i)=fs;

    end
    fprintf('Done detecting events!\n')
    fs = unique(fs_array);
    clear fs_array
    save([directorySave filesep 'event_index'],'indexAll','-v7.3')
    par.thresholdAll=thresholdAll;
    par.thersoldArtifacthAll=thersoldArtifacthAll;

    %% 2. Filter out repeated indices (common noise/global artifacts)
    fprintf('\n\nFinding repeated indices \n')
    [nonRepeatedIndex] = filterRepeatedIndex(indexAll,fs,par); % Uses fs from last microelectrode loaded
    nNonRepeated = numel(cell2mat(nonRepeatedIndex));
    [~, nIndexPerChannel] = cellfun(@size, indexAll);
    total = sum(nIndexPerChannel);
    fprintf('Repeated: %i (%2.2f%%) |  Accepted: %i (%2.2f%%) \n',total-nNonRepeated,100*(1-nNonRepeated/total),nNonRepeated,100*nNonRepeated/total)
    fprintf('Done removing reapeated indices!\n')

    %% 3. Loop through micro channels. Realign spikes. Convert index samples into index mili seconds. Save
    fprintf('\n\nRe-aligning spikes.\n')
    
    filterCutoffs = [par.sort_fmin par.sort_fmax]; % sorting bandpass filter range
    filterOrder = par.sort_order; % soring bandpass filter order
    parfor i = 1:numChannels                                                              %%%%%%%%%%%%%was parfor
        fprintf('[%i/%i] %s.\n',i,numChannels,microLabels{i})
        queryfile=dir([directoryRead filesep microLabels{i} '*.mat']);

%         microFile = load([queryfile.folder filesep queryfile.name])
        objFile = matfile([queryfile.folder filesep queryfile.name]);
        data = objFile.data(1,considered_window.start:considered_window.end);

        [data2,~] =  butterfilter (filterCutoffs,fs,filterOrder,data,'bandpass');
        [spikes,index_tmp] = alignSpikes(data2,nonRepeatedIndex{i},par,biasIndex); 
        index = (index_tmp+biasIndex)/(fs/1e3); %indices in ms
        parsave([directorySave filesep microLabels{i} '_spikes'],spikes,index)
    end
    fprintf('Done re-aligning spikes!\n')  
end


%% %%%%%%%%%%%%%%%%%%% Functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Filter function
function [x,g] =  butterfilter (fc, fs, order, data, type)
    %[x,g] = butterfilter(fc, fs, order, data, type)
    % Zero-phase Butterworth filter
    % Outputs: x is the filtered data
    %          g is the gain of the filter
    % Inputs:  fc = Cut-off frequency (Hz)
    %          fs = Sampling rate (Hz)
    %          order = Filter order
    %          data = data to be filtered
    %          type: 'low', 'high', 'bandpass' or 'stop'
    
    if strcmpi(type,'bandpass')
        [B,A] = butter(order,2*fc/fs);     % Creates two vectors for TS = B/A
    else
        [B,A] = butter(order,2*fc/fs,type);     % Creates two vectors for TS = B/A
    end
    % [sos,g] = tf2sos(B,A);
    x = filtfilt(B,A,data);                 
    [~,g] = tf2sos(B,A);
end

% Save function for parallel loop
function parsave(fname, spikes,index)
    save(fname, 'spikes','index','-v7.3')
end