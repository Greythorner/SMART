function algorithm_for_position()

% =========================================================================


%% ===== User configuration ==================================================
% Sampling
fs                        = 1000;        % Hz, nominal sampling rate
blockSec                  = 4.0;         % seconds per block (fixed 4 s)
blockSamples              = round(blockSec * fs);    % 4000 samples

% Geometry & search
geom.vel                  = 50000/60;    % µm/s, vertical speed
geom.squareSideUm         = 3000;        % 3 mm → 3000 µm
search.gridStepUm         = 250;         % candidate grid step (µm)

% Peak detection
peakDet.minProm           = 5e-2;        % V, minimum peak prominence
peakDet.minPeakDistSec    = 0.2;         % s, minimum peak spacing

% Matching score (time-of-contact alignment)
score.matching_threshold  = 0.05;        % s, nearest-neighbor tolerance
score.sigma               = 0.02;        % s, soft weighting scale

% Visualization
viz.gridN                 = 200;         % dense surface raster for display
viz.fontSize              = 10;
viz.viewAzEl              = [-43 34];
viz.barGridN              = 12;          % N×N bars within square footprint
viz.maxBarHeightUm        = 1000;        % axis z-limit cap for bars (µm)
viz.barEdgeColor          = [0.2 0.2 0.2];
viz.barFaceAlpha          = 0.95;
viz.forceModel            = 'hertz';     % 'hertz' | 'linear'
viz.applyRadialProfile    = true;
viz.radialScale           = 1.0;

% Position & force windowing (adaptive)
posWindowSec_default      = 0.1;         % start with 100 ms for position
posWindowFallbacks        = [0.2 0.3 0.5 1.0];  % fallback ladder (s)
% Effective rule: try 0.1 s; if insufficient peaks/fit, try 0.2/0.3/0.5/1.0 s.

% Serial source (ASCII, one value per line)
ser.port                  = "COM5";      % <-- set your COM port
ser.baud                  = 115200;      % serial baud rate
ser.terminator            = "LF";        % "LF" | "CR" | "CR/LF"
ser.timeoutSec            = 15;          % per-block timeout (s)
ser.flushOnStart          = true;        % flush stale input on start
ser.blocksToRead          = 9999;        % number of blocks to run
ser.pauseBetweenBlocksSec = 0.2;         % optional inter-block pause (s)
ser.holdAfterPredictSec   = 0.5;         % animation duration per block (s)


%% ===== Load surface from base workspace ====================================
if ~evalin('base','exist(''Untitled'',''var'')')
    error('Base workspace must contain table "Untitled" with VarName1/2/3 (x,y,z).');
end
U = evalin('base','Untitled');

surfaceUnits = 'mm'; % 'mm' or 'um'
scale = strcmpi(surfaceUnits,'mm')*1000 + strcmpi(surfaceUnits,'um')*1;

x = double(U.VarName1) * scale;   % µm
y = double(U.VarName2) * scale;   % µm
z = double(U.VarName3) * scale;   % µm
z = z - min(z);                   % shift so min(z)=0

x_min = min(x); x_max = max(x);
y_min = min(y); y_max = max(y);
sensor_max_z = max(z);

Fsurf = scatteredInterpolant(x, y, z, 'natural', 'linear');
[Xg, Yg] = meshgrid(linspace(x_min,x_max,viz.gridN), linspace(y_min,y_max,viz.gridN));
Zs = Fsurf(Xg,Yg); Zs = max(Zs,0);

% Dense candidate points (for grid search & display)
peakX = Xg(:); peakY = Yg(:); peakZ = Zs(:);

% Z axis limit (max of sensor height and bar cap)
z_max = max(sensor_max_z, viz.maxBarHeightUm);


%% ===== UI ================================================================
S = struct(); S.isClosing=false; S.running=false; S.mode='Single';
S.fig = figure('Name','Bmfe Algorithm (Serial Playback)', ...
    'Color','w','Position',[100 60 1200 750], 'IntegerHandle','off');

S.txtMode = uicontrol('Style','text','Units','normalized','Position',[0.02 0.94 0.58 0.04], ...
    'String','', 'Visible','off', 'BackgroundColor','w', ...
    'HorizontalAlignment','left','FontSize',viz.fontSize);

S.modePopup = uicontrol('Style','popupmenu','Units','normalized', ...
    'Position',[0.61 0.94 0.10 0.045], 'FontSize',viz.fontSize, ...
    'String',{'Single','Dual'}, 'Callback',@(~,~)onModeChange());

S.btnStart = uicontrol('Style','pushbutton','String','Start', 'Units','normalized', ...
    'Position',[0.85 0.94 0.10 0.045],'Callback',@startPlayback,'FontSize',viz.fontSize);
S.btnStop  = uicontrol('Style','pushbutton','String','Stop',  'Units','normalized', ...
    'Position',[0.72 0.94 0.10 0.045],'Callback',@stopPlayback,'FontSize',viz.fontSize,'Enable','off');

S.tl = tiledlayout(S.fig,1,2,'TileSpacing','loose','Padding','compact');

% Left panel: surface + growing pressed area
S.ax3D = nexttile(S.tl,1);
surf(S.ax3D,Xg,Yg,Zs,'EdgeColor','none'); hold(S.ax3D,'on');
colormap(S.ax3D,parula); shading(S.ax3D,'interp');
xlabel(S.ax3D,'X (µm)'); ylabel(S.ax3D,'Y (µm)'); zlabel(S.ax3D,'Z (µm)');
axis(S.ax3D,[x_min x_max y_min y_max 0 z_max]);
axis(S.ax3D,'equal'); axis(S.ax3D,'vis3d'); grid(S.ax3D,'on');
set(S.ax3D,'XDir','normal','YDir','normal','ZDir','normal');
view(S.ax3D, viz.viewAzEl);
title(S.ax3D,'Surface + expanding pressed area (1:red, 2:magenta)');
S.pressPts3D_1 = scatter3(S.ax3D,nan,nan,nan,18,'r','filled');
S.pressPts3D_2 = scatter3(S.ax3D,nan,nan,nan,18,[1 0 1],'filled');

% Right panel: square footprint + 3D bars
S.axPos = nexttile(S.tl,2); hold(S.axPos,'on'); grid(S.axPos,'on');
xlabel(S.axPos,'X (µm)'); ylabel(S.axPos,'Y (µm)'); zlabel(S.axPos,'Force height (µm)');
title(S.axPos,'Square (3 mm) + 3D bars (synchronized growth)');
axis(S.axPos,[x_min x_max y_min y_max 0 z_max]);
axis(S.axPos,'equal'); axis(S.axPos,'vis3d');
set(S.axPos,'XDir','normal','YDir','normal','ZDir','normal');
view(S.axPos, viz.viewAzEl);
caxis(S.axPos,[0 1]);
S.posCbar = colorbar(S.axPos,'eastoutside'); S.posCbar.Label.String = 'Force (norm)';

S.rectH1   = rectangle('Parent',S.axPos,'Position',[x_min y_min 1 1], 'EdgeColor',[0 0 0],'LineWidth',2,'Visible','off');
S.centerH1 = plot3(S.axPos,nan,nan,nan,'kx','MarkerSize',10,'LineWidth',2);
S.rectH2   = rectangle('Parent',S.axPos,'Position',[x_min y_min 1 1], 'EdgeColor',[0 0.5 0],'LineWidth',2,'Visible','off');
S.centerH2 = plot3(S.axPos,nan,nan,nan,'x','Color',[0 0.5 0],'MarkerSize',10,'LineWidth',2);

S.depthText   = text(mean([x_min x_max]), y_max, z_max, ...
                     '', 'Parent', S.axPos, 'HorizontalAlignment','center', ...
                     'VerticalAlignment','top','FontSize',viz.fontSize,'Color','k');

S.barPatches1 = []; S.barPatches2 = [];

% Link camera properties across panels
propsToLink = {'CameraPosition','CameraTarget','CameraUpVector','CameraViewAngle', ...
               'XLim','YLim','ZLim','DataAspectRatio','PlotBoxAspectRatio'};
S.camLink = linkprop([S.ax3D, S.axPos], propsToLink);
setappdata(S.fig,'camLink',S.camLink);


%% ===== State =============================================================
S.fs       = fs;
S.blockSec = blockSec;
S.blockSamples = blockSamples;

S.geom     = geom;
S.search   = search;
S.peakDet  = peakDet;
S.score    = score;
S.viz      = viz;

% Windowing state
S.posWindowSec_default = posWindowSec_default;
S.posWindowFallbacks   = posWindowFallbacks;
S.posWindowUsed        = posWindowSec_default;           % actually used “position window”
S.forceWindowSec       = max(blockSec - S.posWindowUsed, 0);  % derived “force window”

S.surface  = struct('Xg',Xg,'Yg',Yg,'Zs',Zs,'x_min',x_min,'x_max',x_max, ...
                    'y_min',y_min,'y_max',y_max,'sensor_max_z',sensor_max_z, ...
                    'z_max', z_max, 'Fsurf', Fsurf);
S.peaks    = struct('X',peakX,'Y',peakY,'Z',peakZ);

S.last = initLast();

% Filtering (detection only)
S.filt.baseline_window = 201;
S.filt.cutoff_freq     = 20;     % Hz
S.filt.butter_order    = 4;

% Serial
S.ser = ser;
S.serialObj = [];   % serialport object

% Counters and flags
S.blockIdx = 0;
S.running  = false;
S.isClosing= false;

S.holdAfterPredictSec = ser.holdAfterPredictSec;

guidata(S.fig,S);
S.fig.CloseRequestFcn = @(h,ev) onClose(h);


%% ===== Callbacks ==========================================================
    function onModeChange()
        fig = gcbf; if isempty(fig), fig = S.fig; end
        S = guidata(fig);
        if S.running
            modes = {'Single','Dual'};
            curIdx = find(strcmpi(S.mode, modes), 1); if isempty(curIdx), curIdx = 1; end
            set(S.modePopup,'Value',curIdx);
            return;
        end
        val  = get(S.modePopup,'Value');
        list = get(S.modePopup,'String');
        if iscell(list), S.mode = list{val};
        elseif isstring(list), S.mode = char(list(val));
        elseif ischar(list), S.mode = strtrim(list(val,:));
        else, S.mode = 'Single'; end
        guidata(fig,S);
    end

    function startPlayback(~,~)
        fig = gcbf; if isempty(fig), fig = S.fig; end
        S = guidata(fig);
        if isstruct(S) && isfield(S,'running') && S.running, return; end

        % Open serial port
        try
            S.serialObj = serialport(S.ser.port, S.ser.baud, 'Timeout', S.ser.timeoutSec);
            configureTerminator(S.serialObj, S.ser.terminator);
            if S.ser.flushOnStart
                flush(S.serialObj);
            end
        catch ME
            errordlg(sprintf('Failed to open serial port %s:\n%s',string(S.ser.port),ME.message),'Serial');
            return;
        end

        % Reset state and UI
        S.running = true; S.isClosing=false; S.blockIdx = 0; S.last = initLast();
        guidata(fig,S); refreshRightPanels(0);
        set(S.btnStart,'Enable','off'); set(S.btnStop,'Enable','on'); set(S.modePopup,'Enable','off');

        % Block loop
        for kBlk = 1:S.ser.blocksToRead
            S = guidata(fig);
            if ~S.running || S.isClosing, break; end

            % Blocking read: 4000 samples per block
            [ok, vRaw] = readOneBlockFromSerial(S.serialObj, S.blockSamples, S.ser.timeoutSec);
            if ~ok
                S.last = initLast();
                S.last.modeTxt = sprintf('Block #%d | Serial timeout or parse error', S.blockIdx+1);
                guidata(fig,S);
                refreshRightPanels(0);
                break;
            end

            % Process one block
            processOneBlock(fig, vRaw(:));

            % Optional pause between blocks
            if S.ser.pauseBetweenBlocksSec > 0
                t0 = tic; 
                while toc(t0) < S.ser.pauseBetweenBlocksSec
                    if ~S.running || S.isClosing, break; end
                    pause(0.02); drawnow;
                end
            end
        end

        stopPlayback();
    end

    function stopPlayback(~,~)
        fig = gcbf; if isempty(fig), fig = S.fig; end
        if ~ishandle(fig), return; end
        S = guidata(fig);
        if ~isstruct(S), return; end

        % Tidy up serial port
        if ~isempty(S.serialObj)
            try, configureCallback(S.serialObj, "off"); catch, end
            try, flush(S.serialObj); catch, end
            try, clearSio(S.serialObj); catch, end
            S.serialObj = [];
        end

        S.running = false; S.isClosing = false; guidata(fig,S);
        set(S.btnStart,'Enable','on'); set(S.btnStop,'Enable','off'); set(S.modePopup,'Enable','on');
        set(S.txtMode,'String','Mode: stopped');
    end

    function onClose(hfig)
        if ~ishandle(hfig), return; end
        S = guidata(hfig);
        if isstruct(S)
            S.isClosing = true; S.running=false; guidata(hfig,S);
            if ~isempty(S.serialObj)
                try, configureCallback(S.serialObj, "off"); catch, end
                try, flush(S.serialObj); catch, end
                try, clearSio(S.serialObj); catch, end
            end
        end
        delete(hfig);
    end

    function processOneBlock(fig, vRaw)
        % Core routine for a single 4-second block with adaptive position window.
        S = guidata(fig);
        if ~isstruct(S) || ~S.running || S.isClosing, return; end

        fs = S.fs; dt = 1/fs;
        tBlock = (0:numel(vRaw)-1)'*dt;

        %% --- Preprocessing over the entire block ---
        bw = S.filt.baseline_window; if mod(bw,2)==0, bw = bw+1; end
        baseline  = movmean(vRaw, bw);
        detrended = vRaw - baseline;

        Wn = S.filt.cutoff_freq/(fs/2);
        Wn = min(max(Wn, 1e-4), 0.9999);
        [b,a] = butter(S.filt.butter_order, Wn, 'low');
        try, vProc = filtfilt(b, a, detrended); catch, vProc = filter(b, a, detrended); end

        %% --- Adaptive position-window trials (early portion only) ---
        try_windows = [S.posWindowSec_default, S.posWindowFallbacks];
        try_windows = try_windows(try_windows < S.blockSec - 0.01); % leave time for force window

        gotCenter = false;
        usedWin   = S.posWindowSec_default;
        modeLabel = 'Final (Single)';

        for w = try_windows
            % Peak detection restricted to [0, w]
            posMask = (tBlock <= w);
            vACpos  = vProc(posMask);
            tPos    = tBlock(posMask);

            sigmaV  = 1.4826 * mad(vACpos, 1);
            minProm = max([S.peakDet.minProm, 3*sigmaV, 1e-5]);
            minDistSamp = max(1, round(S.peakDet.minPeakDistSec * fs));
            minWidth    = max(1, round(0.002 * fs));

            [~, locsAC] = findpeaks(vACpos, ...
                'MinPeakProminence', minProm, ...
                'MinPeakDistance',   minDistSamp, ...
                'MinPeakWidth',      minWidth);

            edgeMs = 0.02;
            keep   = (tPos(locsAC) > edgeMs) & (tPos(locsAC) < w - edgeMs);
            locs   = locsAC(keep);

            tk = sort(tPos(locs));
            if isempty(tk)
                % No peaks found for this window; try a longer one
                continue;
            end

            % Position solve (Single or Dual)
            if strcmpi(S.mode,'Single')
                adj_res = tk - tk(1);
                [center1, idx1, tpred1] = estimateSquareCenterFromPeaks( ...
                    adj_res, S.peaks.X, S.peaks.Y, S.peaks.Z, ...
                    S.geom.squareSideUm, ...
                    [S.surface.x_min S.surface.x_max S.surface.y_min S.surface.y_max], ...
                    S.search.gridStepUm, S.surface.sensor_max_z, S.geom.vel, S.score);

                tmp = buildOne(center1, idx1, tpred1, S, 'Final (Single)');
                if tmp.valid1
                    S.last = tmp; modeLabel = 'Final (Single)'; gotCenter = true; usedWin = w; break;
                end
            else
                obs_rel_full = tk - tk(1);
                [c1, idx1, tp1] = estimateSquareCenterFromPeaks( ...
                    obs_rel_full, S.peaks.X, S.peaks.Y, S.peaks.Z, ...
                    S.geom.squareSideUm, ...
                    [S.surface.x_min S.surface.x_max S.surface.y_min S.surface.y_max], ...
                    S.search.gridStepUm, S.surface.sensor_max_z, S.geom.vel, S.score);
                obs_remain = greedyRemoveMatched(obs_rel_full, tp1, S.score.matching_threshold);

                [X2,Y2,Z2,maskLocalToGlobal] = dropInsideSquare(S.peaks.X, S.peaks.Y, S.peaks.Z, c1, S.geom.squareSideUm);
                useTimes = obs_remain; if numel(useTimes)<2, useTimes = obs_rel_full; end

                if numel(X2) >= 3
                    [c2, idx2_local, tp2] = estimateSquareCenterFromPeaks( ...
                        useTimes, X2, Y2, Z2, ...
                        S.geom.squareSideUm, ...
                        [S.surface.x_min S.surface.x_max S.surface.y_min S.surface.y_max], ...
                        S.search.gridStepUm, S.surface.sensor_max_z, S.geom.vel, S.score);
                    idx2 = localToGlobalIndex(idx2_local, X2, Y2, Z2, S.peaks.X, S.peaks.Y, S.peaks.Z, maskLocalToGlobal);
                else
                    c2=[nan nan]; idx2=[]; tp2=[];
                end
                tStart2 = 0; if ~isempty(useTimes), tStart2 = min(useTimes); end

                tmp = buildTwo(c1, idx1, tp1, c2, idx2, tp2, tStart2, S, 'Final (Dual)');
                if tmp.valid1 || (isfield(tmp,'valid2') && tmp.valid2)
                    S.last = tmp; modeLabel = 'Final (Dual)'; gotCenter = true; usedWin = w; break;
                end
            end
        end

        % If still no valid solution, report and clear overlays
        if ~gotCenter
            S.last = initLast();
            S.last.modeTxt = sprintf('Block #%d | No peaks/center up to %.0f ms', S.blockIdx+1, try_windows(end)*1000);
            guidata(fig,S);
            refreshRightPanels(0);
        else
            % Success: update windows and annotate
            S.posWindowUsed   = usedWin;
            S.forceWindowSec  = max(S.blockSec - S.posWindowUsed, 0);
            guidata(fig,S);

            if strcmpi(S.mode,'Single')
                S.last.modeTxt = sprintf('%s | posWin=%.0f ms, forceWin=%.0f ms', ...
                    modeLabel, usedWin*1000, S.forceWindowSec*1000);
            else
                if S.last.valid2
                    S.last.modeTxt = sprintf('%s (both) | posWin=%.0f ms, forceWin=%.0f ms', ...
                        modeLabel, usedWin*1000, S.forceWindowSec*1000);
                else
                    S.last.modeTxt = sprintf('%s (only #1) | posWin=%.0f ms, forceWin=%.0f ms', ...
                        modeLabel, usedWin*1000, S.forceWindowSec*1000);
                end
            end
            guidata(fig,S);

            % Refresh & animate (bar growth synchronized to force window)
            refreshRightPanels(0);
            animatePressSync(fig);
        end

        % Advance block counter
        S.blockIdx = S.blockIdx + 1;
        guidata(fig,S);
        drawnow limitrate;
    end

end % ===== end main =====


%% ===== Serial utilities ====================================================
function [ok, vRaw] = readOneBlockFromSerial(sp, blockSamples, timeoutSec)
% Blocking read of one scalar-valued block from a serialport stream.
v = nan(blockSamples,1);
i = 1;
tStart = tic;
while i <= blockSamples
    if toc(tStart) > timeoutSec
        ok = false; vRaw = []; return;
    end
    try
        if sp.NumBytesAvailable > 0
            line = strtrim(readline(sp));
            val = str2double(line);
            if ~isfinite(val), continue; end
            v(i) = val; i = i + 1;
        else
            pause(0.001);
        end
    catch
        pause(0.001);
    end
end
ok = true; vRaw = v;
end

function clearSio(sp)
% Best-effort serial cleanup
try, fclose(instrfind); catch, end %#ok<INSTRF>
try, delete(sp); catch, end
end


%% ===== Helper structs ======================================================
function L = initLast()
% Container for the most recent solve/visualization payload
L = struct(... 
    'valid1',false,'center1',[nan nan],'rect1',[nan nan nan nan], ...
    'pressedIdx1',[],'insideIdx1',[],'insideForce1',[], ...
    'valid2',false,'center2',[nan nan],'rect2',[nan nan nan nan], ...
    'pressedIdx2',[],'insideIdx2',[],'insideForce2',[], ...
    'Xc1',[],'Yc1',[],'Fn1',[],'Xc2',[],'Yc2',[],'Fn2',[], ...
    'dx',[],'dy',[], 'modeTxt','', ...
    'tInside1',[],'tInside2',[], ...
    'tStart1',NaN,'tStart2',NaN);
end

function L = buildOne(center, idxInside, tpredInside, S, modeLabel)
% Pack single-square solution
L = initLast();
if isempty(idxInside) || ~all(isfinite(center))
    L.modeTxt = sprintf('%s | No valid result', modeLabel); return;
end
[Xc, Yc, Fn, dx, dy] = computeBarForces(center, S, 0);

half = S.geom.squareSideUm/2; cx = center(1); cy = center(2);
pressedNowMask = false(numel(S.peaks.X),1);
pressedNow = idxInside(tpredInside <= S.blockSec + eps);
pressedNowMask(pressedNow) = true;

L.valid1       = true;
L.center1      = center;
L.rect1        = [cx-half, cy-half, S.geom.squareSideUm, S.geom.squareSideUm];
L.pressedIdx1  = find(pressedNowMask);
L.insideIdx1   = idxInside(:);
L.insideForce1 = [];

L.Xc1 = Xc; L.Yc1 = Yc; L.Fn1 = Fn; L.dx = dx; L.dy = dy;
L.tInside1 = tpredInside(:);
L.tStart1  = 0;
L.modeTxt  = sprintf('%s', modeLabel);
end

function L = buildTwo(c1, idx1, tp1, c2, idx2, tp2, tStart2, S, modeLabel)
% Pack dual-square solution
L = buildOne(c1, idx1, tp1, S, [modeLabel ' #1']);
if isempty(idx2) || ~all(isfinite(c2))
    L.modeTxt = sprintf('%s (only #1 valid)', modeLabel);
    return;
end
[Xc2, Yc2, Fn2, dx, dy] = computeBarForces(c2, S, max(0,tStart2));

half = S.geom.squareSideUm/2; cx2 = c2(1); cy2 = c2(2);
pressedNowMask2 = false(numel(S.peaks.X),1);
pressedNow2 = idx2(tp2 <= S.blockSec + eps);
pressedNowMask2(pressedNow2) = true;

L.valid2       = true;
L.center2      = c2;
L.rect2        = [cx2-half, cy2-half, S.geom.squareSideUm, S.geom.squareSideUm];
L.pressedIdx2  = find(pressedNowMask2);
L.insideIdx2   = idx2(:);
L.insideForce2 = [];

L.Xc2 = Xc2; L.Yc2 = Yc2; L.Fn2 = Fn2; L.dx = dx; L.dy = dy;
L.tInside2 = tp2(:);
L.tStart2  = tStart2;
end


%% ===== Solvers & utilities ===============================================
function [center, idxInside, tpredInside] = estimateSquareCenterFromPeaks(adj_res_times, X, Y, Z, L, bounds, stepUm, sensor_max_z, vel, score)
% Grid-search center of an L×L square by matching predicted vs observed
% relative contact times (nearest-neighbor matching with soft weights).
center = [nan nan]; idxInside = []; tpredInside = [];
if isempty(adj_res_times) || isempty(X), return; end

adj_res_times = double(adj_res_times(:));
x_min = bounds(1)+L/2; x_max = bounds(2)-L/2;
y_min = bounds(3)+L/2; y_max = bounds(4)-L/2;
if x_min>=x_max || y_min>=y_max, return; end

xcand = x_min:stepUm:x_max;
ycand = y_min:stepUm:ymax;

bestScore = -inf; bestCenter = [nan nan]; bestIdx = []; bestTP = [];

for cx = xcand
    for cy = ycand
        half = L/2;
        mask = (X>=cx-half & X<=cx+half & Y>=cy-half & Y<=cy+half);
        I = find(mask);
        if numel(I) < 3, continue; end

        t_pred_abs = (sensor_max_z - Z(I)) / vel;
        t_pred_rel = t_pred_abs - min(t_pred_abs);

        m = numel(t_pred_rel);
        dt = zeros(m,1);
        for j = 1:m
            dt(j) = min(abs(adj_res_times - t_pred_rel(j)));
        end
        w = exp(-dt/score.sigma) .* (dt <= score.matching_threshold);
        s = sum(w) / max(m,1);

        if s > bestScore
            bestScore = s; bestCenter = [cx cy]; bestIdx = I; bestTP = t_pred_rel;
        end
    end
end

center       = bestCenter;
idxInside    = bestIdx;
tpredInside  = bestTP;
end

function obs_remain = greedyRemoveMatched(obs_times, pred_times, thr)
% Greedily remove observed times that match predicted times within thr.
obs = sort(obs_times(:));
if isempty(obs) || isempty(pred_times), obs_remain = obs; return; end
used = false(size(obs));
pr  = sort(pred_times(:));
for k=1:numel(pr)
    [md, idx] = min(abs(obs - pr(k)));
    if ~isempty(idx) && md <= thr && ~used(idx)
        used(idx) = true;
    end
end
obs_remain = obs(~used);
end

function [X2,Y2,Z2,maskLocalToGlobal] = dropInsideSquare(X,Y,Z, center, L)
% Remove points inside the first square to search for a second square.
if any(~isfinite(center))
    X2=X;Y2=Y;Z2=Z; maskLocalToGlobal = true(size(X));
    return;
end
half = L/2; cx=center(1); cy=center(2);
keep = ~(X>=cx-half & X<=cx+half & Y>=cy-half & Y<=cy+half);
X2 = X(keep); Y2 = Y(keep); Z2 = Z(keep);
maskLocalToGlobal = keep;
end

function idxGlobal = localToGlobalIndex(idxLocal, X2, Y2, Z2, Xall, Yall, Zall, maskLocalToGlobal)
% Map indices in the reduced (outer) set back to the full set.
if isempty(idxLocal), idxGlobal = []; return; end
tol = 1e-9;
idxGlobal = zeros(size(idxLocal));
mapLocalToGlobal = find(maskLocalToGlobal);
for i=1:numel(idxLocal)
    gi = mapLocalToGlobal(idxLocal(i));
    if abs(Xall(gi)-X2(idxLocal(i)))<tol && abs(Yall(gi)-Y2(idxLocal(i)))<tol && abs(Zall(gi)-Z2(idxLocal(i)))<tol
        idxGlobal(i) = gi;
    else
        xi=X2(idxLocal(i)); yi=Y2(idxLocal(i)); zi=Z2(idxLocal(i));
        [~,cand] = min( (Xall-xi).^2 + (Yall-yi).^2 + (Zall-zi).^2 );
        idxGlobal(i) = cand;
    end
end
end

function hidePatches(patches)
% Hide a group of patch handles (if any exist).
if isempty(patches), return; end
for k=1:numel(patches)
    if isgraphics(patches(k)), set(patches(k),'Visible','off'); end
end
end

function patches = drawOrUpdateBars(ax, patches, Xc, Yc, F, dx, dy, hmax, vz, cmapArr)
% Create or update the 3D bar field over the square.
N = numel(F);
if isempty(patches) || numel(patches) ~= N*5 || any(~isgraphics(patches))
    if ~isempty(patches)
        try, delete(patches(isgraphics(patches))); catch, end
    end
    patches = gobjects(N*5,1);
    for i=1:N
        [Vtops, V1, V2, V3, V4, fc] = barVerticesColors(i,Xc,Yc,F,dx,dy,hmax,cmapArr);
        patches(5*(i-1)+1) = patch(ax,'Faces',[1 2 3 4],'Vertices',Vtops,'FaceColor',fc,'EdgeColor',vz.barEdgeColor,'FaceAlpha',vz.barFaceAlpha);
        patches(5*(i-1)+2) = patch(ax,'Faces',[1 2 3 4],'Vertices',V1,   'FaceColor',fc,'EdgeColor',vz.barEdgeColor,'FaceAlpha',vz.barFaceAlpha);
        patches(5*(i-1)+3) = patch(ax,'Faces',[1 2 3 4],'Vertices',V2,   'FaceColor',fc,'EdgeColor',vz.barEdgeColor,'FaceAlpha',vz.barFaceAlpha);
        patches(5*(i-1)+4) = patch(ax,'Faces',[1 2 3 4],'Vertices',V3,   'FaceColor',fc,'EdgeColor',vz.barEdgeColor,'FaceAlpha',vz.barFaceAlpha);
        patches(5*(i-1)+5) = patch(ax,'Faces',[1 2 3 4],'Vertices',V4,   'FaceColor',fc,'EdgeColor',vz.barEdgeColor,'FaceAlpha',vz.barFaceAlpha);
    end
    return;
end

for i=1:N
    [Vtops, V1, V2, V3, V4, fc] = barVerticesColors(i,Xc,Yc,F,dx,dy,hmax,cmapArr);
    set(patches(5*(i-1)+1),'Vertices',Vtops,'FaceColor',fc,'Visible','on');
    set(patches(5*(i-1)+2),'Vertices',V1,   'FaceColor',fc,'Visible','on');
    set(patches(5*(i-1)+3),'Vertices',V2,   'FaceColor',fc,'Visible','on');
    set(patches(5*(i-1)+4),'Vertices',V3,   'FaceColor',fc,'Visible','on');
    set(patches(5*(i-1)+5),'Vertices',V4,   'FaceColor',fc,'Visible','on');
end
end

function [Vtops,V1,V2,V3,V4,fc] = barVerticesColors(i,Xc,Yc,F,dx,dy,hmax,cmapArr)
% Compute vertices and face color for a single bar cell.
x0 = Xc(i)-dx/2; x1 = Xc(i)+dx/2;
y0 = Yc(i)-dy/2; y1 = Yc(i)+dy/2;
f  = max(0,min(1,F(i)));
h  = f * hmax;
cidx = max(1,min(size(cmapArr,1), 1+floor(f*(size(cmapArr,1)-1)) ));
fc   = cmapArr(cidx,:);
Vtops = [x0 y0 h;  x1 y0 h;  x1 y1 h;  x0 y1 h];
V1 = [x0 y0 0;  x1 y0 0;  x1 y0 h;  x0 y0 h];
V2 = [x1 y0 0;  x1 y1 0;  x1 y1 h;  x1 y0 h];
V3 = [x1 y1 0;  x0 y1 0;  x0 y1 h;  x1 y1 h];
V4 = [x0 y1 0;  x0 y0 0;  x0 y0 h;  x0 y1 h];
end


%% ===== Force field (normalized) ===========================================
function [Xc, Yc, Fn, dx, dy] = computeBarForces(center, S, startOffsetSec)
% Compute normalized force field over the square footprint.
half = S.geom.squareSideUm/2; cx = center(1); cy = center(2);
N  = S.viz.barGridN; dx = S.geom.squareSideUm / N; dy = S.geom.squareSideUm / N;
[Xc, Yc] = meshgrid( (cx-half)+dx/2:dx:(cx+half)-dx/2, ...
                     (cy-half)+dy/2:dy:(cy+half)-dy/2 );
Xc = Xc(:); Yc = Yc(:);

% Surface height at bar centers (robust fallback)
Zc = S.surface.Fsurf(Xc, Yc);
bad = ~isfinite(Zc);
if any(bad)
    Zc(bad) = griddata(S.surface.Xg, S.surface.Yg, S.surface.Zs, Xc(bad), Yc(bad), 'nearest');
    Zc(~isfinite(Zc)) = median(S.surface.Zs(:),'omitnan');
end

% Predicted relative arrival (lower regions touch earlier)
t_pred_abs = (S.surface.sensor_max_z - Zc) / S.geom.vel;
t_rel = t_pred_abs - min(t_pred_abs);
offset = max(0, startOffsetSec);

% Force growth window = blockSec − actual position window
ageSec = S.forceWindowSec;

% Penetration depth proxy (µm)
depth_um = S.geom.vel * max(0, ageSec - (t_rel + offset));

% Force law
switch lower(S.viz.forceModel)
    case 'hertz'
        Fn = depth_um .^ 1.5;
    otherwise
        Fn = depth_um;
end

% Optional radial taper
if isfield(S.viz,'applyRadialProfile') && S.viz.applyRadialProfile
    R = (S.geom.squareSideUm/2) * max(0.1, S.viz.radialScale);
    r = hypot(Xc-cx, Yc-cy);
    w = zeros(size(r)); in = r<=R;
    w(in) = sqrt(max(0, 1 - (r(in)/max(R,eps)).^2));
    Fn = Fn .* w;
end

% Normalize to [0, 1]
mx = max(Fn);
if ~isfinite(mx) || mx<=0, Fn(:)=0; else, Fn = max(0, min(1, Fn./mx)); end
end


%% ===== Right panel refresh (scale drives bar growth) ======================
function refreshRightPanels(scale)
if nargin<1, scale = 1; end
S = guidata(gcf);

% Clear left-panel pressed points (animation controls these)
set(S.pressPts3D_1,'XData',nan,'YData',nan,'ZData',nan);
set(S.pressPts3D_2,'XData',nan,'YData',nan,'ZData',nan);

% Hide/clear squares & centers
set(S.rectH1,'Visible','off'); set(S.rectH2,'Visible','off');
set(S.centerH1,'XData',nan,'YData',nan,'ZData',nan);
set(S.centerH2,'XData',nan,'YData',nan,'ZData',nan);

% Square #1
if S.last.valid1 && all(isfinite(S.last.rect1))
    set(S.rectH1,'Position',S.last.rect1,'Visible','on');
    set(S.centerH1,'XData',S.last.center1(1),'YData',S.last.center1(2),'ZData',0);

    Fn1 = S.last.Fn1 .* max(0,min(1,scale)); % synchronized bar growth
    S.barPatches1 = drawOrUpdateBars(S.axPos, S.barPatches1, ...
        S.last.Xc1, S.last.Yc1, Fn1, S.last.dx, S.last.dy, ...
        S.viz.maxBarHeightUm, S.viz, hot(256));
else
    hidePatches(S.barPatches1);
end

% Square #2 (dual mode)
if isfield(S.last,'valid2') && S.last.valid2 && all(isfinite(S.last.rect2))
    set(S.rectH2,'Position',S.last.rect2,'Visible','on');
    set(S.centerH2,'XData',S.last.center2(1),'YData',S.last.center2(2),'ZData',0);

    Fn2 = S.last.Fn2 .* max(0,min(1,scale));
    S.barPatches2 = drawOrUpdateBars(S.axPos, S.barPatches2, ...
        S.last.Xc2, S.last.Yc2, Fn2, S.last.dx, S.last.dy, ...
        S.viz.maxBarHeightUm, S.viz, winter(256));
else
    hidePatches(S.barPatches2);
end

% Status text
if isfield(S.last,'modeTxt')
    S.depthText.String = S.last.modeTxt;
else
    S.depthText.String = '';
end

guidata(S.fig,S);
end


%% ===== Synchronized animation ============================================
function animatePressSync(fig)
S = guidata(fig);
if ~isstruct(S) || ~S.running || S.isClosing, return; end

dur = max(0, S.holdAfterPredictSec);
if dur <= 0
    refreshRightPanels(1);
    resetVisualizationToBaseline(fig);
    return;
end

fps   = 30;
steps = max(ceil(dur*fps),1);
dt    = dur/steps;

% Normalize by max predicted arrival times to drive left-panel growth
tmax1 = 0; tmax2 = 0;
if S.last.valid1 && ~isempty(S.last.tInside1), tmax1 = max(S.last.tInside1); end
if isfield(S.last,'valid2') && S.last.valid2 && ~isempty(S.last.tInside2), tmax2 = max(S.last.tInside2); end
if ~isfinite(tmax1) || tmax1<=0, tmax1 = 1; end
if ~isfinite(tmax2) || tmax2<=0, tmax2 = 1; end

for k = 0:steps
    if ~S.running || S.isClosing, return; end
    progress = k/steps;

    % Right panel: bar growth
    refreshRightPanels(progress);

    % Left panel: pressed points expand according to predicted relative times
    if S.last.valid1 && ~isempty(S.last.insideIdx1) && ~isempty(S.last.tInside1)
        sel1 = (S.last.tInside1./tmax1) <= progress + eps;
        idx1 = S.last.insideIdx1(sel1);
        set(S.pressPts3D_1,'XData',S.peaks.X(idx1),'YData',S.peaks.Y(idx1),'ZData',S.peaks.Z(idx1),'Visible','on');
    else
        set(S.pressPts3D_1,'XData',nan,'YData',nan,'ZData',nan);
    end
    if isfield(S.last,'valid2') && S.last.valid2 && ~isempty(S.last.insideIdx2) && ~isempty(S.last.tInside2)
        sel2 = (S.last.tInside2./tmax2) <= progress + eps;
        idx2 = S.last.insideIdx2(sel2);
        set(S.pressPts3D_2,'XData',S.peaks.X(idx2),'YData',S.peaks.Y(idx2),'ZData',S.peaks.Z(idx2),'Visible','on');
    else
        set(S.pressPts3D_2,'XData',nan,'YData',nan,'ZData',nan);
    end

    drawnow limitrate;
    pause(dt);
end

resetVisualizationToBaseline(fig);
end


%% ===== Reset overlays to baseline ========================================
function resetVisualizationToBaseline(fig)
S = guidata(fig);
if ~isstruct(S), return; end

S.last = initLast();

% Left-panel pressed points → cleared (surface remains)
if isgraphics(S.pressPts3D_1)
    set(S.pressPts3D_1,'XData',nan,'YData',nan,'ZData',nan,'Visible','on');
end
if isgraphics(S.pressPts3D_2)
    set(S.pressPts3D_2,'XData',nan,'YData',nan,'ZData',nan,'Visible','on');
end

% Right-panel overlays → hidden/cleared
if isgraphics(S.rectH1),   set(S.rectH1,'Visible','off'); end
if isgraphics(S.rectH2),   set(S.rectH2,'Visible','off'); end
if isgraphics(S.centerH1), set(S.centerH1,'XData',nan,'YData',nan,'ZData',nan); end
if isgraphics(S.centerH2), set(S.centerH2,'XData',nan,'YData',nan,'ZData',nan); end
hidePatches(S.barPatches1);
hidePatches(S.barPatches2);
if isgraphics(S.depthText), S.depthText.String = ''; end

guidata(S.fig,S);
drawnow;
end
