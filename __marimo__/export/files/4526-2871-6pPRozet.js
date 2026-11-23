
    function render({ model, el }) {
        el.innerHTML = `
        <div class="animator-box">
            <div class="button-row">
                <button class="animator-btn main-btn"></button>
                <button class="animator-btn restart-btn" style="display: none;">↻ Restart</button>
            </div>
            <div class="progress-row">
                <div class="animator-track">
                    <div class="animator-fill"></div>
                </div>
                <div class="animator-text">0/0</div>
            </div>
        </div>
        `;

        const mainBtn = el.querySelector('.main-btn');
        const restartBtn = el.querySelector('.restart-btn');
        const text = el.querySelector('.animator-text');
        const fill = el.querySelector('.animator-fill');

        let playing = false;
        let timer = null;

        function getLabel(labelType) {
            const label = model.get(labelType);
            if (labelType === 'label_start') return label || 'Play';
            if (labelType === 'label_stop') return label || 'Stop';
            if (labelType === 'label_restart') return label || 'Restart';
            return label || '';
        }

        function updateButtons() {
            const step = model.get('current_step');
            const ticks = model.get('ticks') || [];
            const total = ticks.length;
            const isComplete = step >= total && total > 0;

            if (isComplete && !playing) {
                // Animation complete - show restart button
                mainBtn.style.display = 'none';
                restartBtn.style.display = 'block';
            } else {
                // Animation in progress or not started
                mainBtn.style.display = 'block';
                restartBtn.style.display = 'none';
                mainBtn.textContent = playing ? ('⏹ ' + getLabel('label_stop')) : ('▶ ' + getLabel('label_start'));
                mainBtn.classList.toggle('stop', playing);
            }
        }

        function update() {
            const step = model.get('current_step');
            const ticks = model.get('ticks') || [];
            const total = ticks.length;
            const pct = total > 0 ? (step / total) * 100 : 0;

            fill.style.width = pct + '%';
            text.textContent = `${step}/${total}`;
            updateButtons();
        }

        function tick() {
            const ticks = model.get('ticks') || [];
            const total = ticks.length;
            let step = model.get('current_step');

            if (!playing || step >= total) {
                playing = false;
                updateButtons();
                return;
            }

            // next step
            step = step + 1;
            model.set('current_step', step);
            model.save_changes();
            update();

            // schedule next
            if (step < total) {
                timer = setTimeout(tick, ticks[step - 1] || 500);
            } else {
                playing = false;
                updateButtons();
            }
        }

        function startAnimation() {
            const total = (model.get('ticks') || []).length;
            if (model.get('current_step') >= total) {
                model.set('current_step', 0);
                model.save_changes();
                update();
            }
            playing = true;
            updateButtons();
            tick();
        }

        function stopAnimation() {
            playing = false;
            clearTimeout(timer);
            updateButtons();
        }

        function restartAnimation() {
            model.set('current_step', 0);
            model.save_changes();
            update();
            startAnimation();
        }

        // Main button click handler (play/stop)
        mainBtn.onclick = () => {
            if (playing) {
                stopAnimation();
            } else {
                startAnimation();
            }
        };

        // Restart button click handler
        restartBtn.onclick = () => {
            restartAnimation();
        };

        model.on('change:current_step', update);
        model.on('change:ticks', update);
        model.on('change:label_start', update);
        model.on('change:label_stop', update);
        model.on('change:label_restart', update);
        update();

        return () => {
            playing = false;
            clearTimeout(timer);
        };
    }

    export default { render };
    