/**
 * 中文文本情感分析系统 - 前端交互脚本（反馈闭环增强版）
 * 功能：批量分析、单句分析、智能客服、图表展示、用户纠错反馈闭环
 * 更新：整合反馈提交、批量行动态绑定、单句反馈、自动触发重训
 */

// ========== 全局变量 ==========
let pieChart = null;
let barChart = null;
let currentBatchResult = null;
let singleResultData = null;          // 存储单句分析结果，供反馈使用

// ========== 工具函数 ==========
function emojiToLabel(emojiOrText) {
    if (typeof emojiOrText !== 'string') return 'neutral';
    if (emojiOrText.includes('😊') || emojiOrText.includes('正面')) return 'positive';
    if (emojiOrText.includes('😠') || emojiOrText.includes('负面')) return 'negative';
    if (emojiOrText.includes('😐') || emojiOrText.includes('中性')) return 'neutral';
    return 'neutral';
}

function getSentimentEmoji(label) {
    const map = { positive: '😊', negative: '😠', neutral: '😐' };
    return map[label] || '😐';
}

function getSentimentText(label) {
    const map = { positive: '正面', negative: '负面', neutral: '中性' };
    return map[label] || '中性';
}

// ========== 反馈提交核心函数 ==========
function submitFeedback(text, predictedLabel, confidence, correctedLabel, callback) {
    fetch('/api/feedback', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            text: text,
            predicted_label: predictedLabel,
            corrected_label: correctedLabel,
            confidence: confidence
        })
    })
    .then(res => res.json())
    .then(data => {
        if (data.success) {
            if (callback) callback(true, data);
        } else {
            if (callback) callback(false, data);
        }
    })
    .catch(err => {
        console.error('反馈提交失败:', err);
        if (callback) callback(false, { message: err.message });
    });
}

// ========== 页面初始化 ==========
document.addEventListener('DOMContentLoaded', function() {
    initTabs();
    initBatchAnalysis();
    initSingleAnalysis();
    initChat();
    initCharts();
    initFeedbackGlobals();   // 全局反馈监听（单句）
});

// ==================== 标签页切换 ====================
function initTabs() {
    const tabs = document.querySelectorAll('.tab');
    const panels = document.querySelectorAll('.tab-panel');

    tabs.forEach(tab => {
        tab.addEventListener('click', function() {
            const targetTab = this.dataset.tab;
            tabs.forEach(t => t.classList.remove('active'));
            this.classList.add('active');
            panels.forEach(p => p.classList.remove('active'));
            document.getElementById(targetTab + '-panel').classList.add('active');
            if (targetTab === 'batch') {
                resizeCharts();
            }
        });
    });
}

// ==================== 批量分析 ====================
function initBatchAnalysis() {
    const uploadArea = document.getElementById('upload-area');
    const fileInput = document.getElementById('file-input');
    const uploadBtn = document.getElementById('upload-btn');
    const exportBtn = document.getElementById('export-btn');

    uploadArea.addEventListener('click', () => fileInput.click());
    uploadBtn.addEventListener('click', () => fileInput.click());

    uploadArea.addEventListener('dragover', (e) => {
        e.preventDefault();
        uploadArea.classList.add('dragover');
    });
    uploadArea.addEventListener('dragleave', () => {
        uploadArea.classList.remove('dragover');
    });
    uploadArea.addEventListener('drop', (e) => {
        e.preventDefault();
        uploadArea.classList.remove('dragover');
        if (e.dataTransfer.files.length > 0) {
            handleFile(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener('change', (e) => {
        if (e.target.files.length > 0) {
            handleFile(e.target.files[0]);
        }
    });

    exportBtn.addEventListener('click', exportResult);
}

function handleFile(file) {
    if (!file.name.endsWith('.csv')) {
        alert('请上传CSV格式的文件！');
        return;
    }

    document.getElementById('upload-area').parentElement.style.display = 'none';
    document.getElementById('loading-section').style.display = 'block';
    document.getElementById('result-section').style.display = 'none';

    let progress = 0;
    const progressEl = document.getElementById('loading-progress');
    const progressInterval = setInterval(() => {
        progress += Math.random() * 15;
        if (progress > 90) progress = 90;
        progressEl.textContent = Math.floor(progress) + '%';
    }, 200);

    const formData = new FormData();
    formData.append('file', file);

    fetch('/api/batch_sentiment', {
        method: 'POST',
        body: formData
    })
    .then(response => response.json())
    .then(data => {
        clearInterval(progressInterval);
        progressEl.textContent = '100%';

        if (data.success) {
            currentBatchResult = data.data;
            setTimeout(() => {
                showBatchResult(data.data);
            }, 300);
        } else {
            alert('分析失败：' + (data.message || '未知错误'));
            resetUpload();
        }
    })
    .catch(error => {
        clearInterval(progressInterval);
        console.error('批量分析错误:', error);
        alert('分析失败：' + error.message);
        resetUpload();
    });
}

function showBatchResult(data) {
    document.getElementById('loading-section').style.display = 'none';
    document.getElementById('result-section').style.display = 'block';

    document.getElementById('stat-total').textContent = data.total;
    document.getElementById('stat-positive').textContent = data.sentiment_counts.positive;
    document.getElementById('stat-negative').textContent = data.sentiment_counts.negative;
    document.getElementById('stat-neutral').textContent = data.sentiment_counts.neutral;

    updatePieChart(data.sentiment_counts);
    updateBarChart(data.sentiment_counts, data.total);
    updateKeywords(data);
    updatePreviewTable(data.preview);

    // 滚动到结果区域
    document.getElementById('result-section').scrollIntoView({ behavior: 'smooth' });
}

function resetUpload() {
    document.getElementById('upload-area').parentElement.style.display = 'block';
    document.getElementById('loading-section').style.display = 'none';
    document.getElementById('result-section').style.display = 'none';
    document.getElementById('file-input').value = '';
}

function exportResult() {
    if (!currentBatchResult || !currentBatchResult.preview) {
        alert('没有可导出的数据');
        return;
    }

    const headers = ['序号', '文本内容', '情感标签', '置信度'];
    const rows = currentBatchResult.preview.map((item, index) => {
        return [index + 1, `"${item.text.replace(/"/g, '""')}"`, item.sentiment_cn, item.confidence.toFixed(4)];
    });

    const csvContent = [headers.join(','), ...rows.map(r => r.join(','))].join('\n');
    const blob = new Blob(['\ufeff' + csvContent], { type: 'text/csv;charset=utf-8;' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = '情感分析结果_' + new Date().toISOString().slice(0, 10) + '.csv';
    link.click();
}

// ==================== 图表相关 ====================
function initCharts() {
    const pieChartDom = document.getElementById('pie-chart');
    if (pieChartDom) {
        pieChart = echarts.init(pieChartDom);
    }
    const barChartDom = document.getElementById('bar-chart');
    if (barChartDom) {
        barChart = echarts.init(barChartDom);
    }
    window.addEventListener('resize', resizeCharts);
}

function resizeCharts() {
    if (pieChart) pieChart.resize();
    if (barChart) barChart.resize();
}

function updatePieChart(counts) {
    if (!pieChart) return;
    const option = {
        tooltip: { trigger: 'item', formatter: '{b}: {c} ({d}%)' },
        legend: { orient: 'vertical', left: 'left', top: 'center' },
        series: [{
            name: '情感分布',
            type: 'pie',
            radius: ['40%', '70%'],
            avoidLabelOverlap: false,
            itemStyle: { borderRadius: 10, borderColor: '#fff', borderWidth: 2 },
            label: { show: false, position: 'center' },
            emphasis: { label: { show: true, fontSize: 20, fontWeight: 'bold' } },
            labelLine: { show: false },
            data: [
                { value: counts.positive, name: '正面😊', itemStyle: { color: '#52c41a' } },
                { value: counts.negative, name: '负面😠', itemStyle: { color: '#ff4d4f' } },
                { value: counts.neutral, name: '中性😐', itemStyle: { color: '#1890ff' } }
            ]
        }]
    };
    pieChart.setOption(option);
}

function updateBarChart(counts, total) {
    if (!barChart) return;
    total = total || 1;
    const option = {
        tooltip: {
            trigger: 'axis',
            axisPointer: { type: 'shadow' },
            formatter: function(params) {
                const item = params[0];
                const percent = ((item.value / total) * 100).toFixed(1);
                return item.name + '<br/>数量: ' + item.value + '<br/>占比: ' + percent + '%';
            }
        },
        grid: { left: '3%', right: '4%', bottom: '3%', containLabel: true },
        xAxis: { type: 'category', data: ['正面😊', '负面😠', '中性😐'], axisLabel: { fontSize: 14 } },
        yAxis: { type: 'value', name: '数量' },
        series: [{
            name: '数量',
            type: 'bar',
            barWidth: '50%',
            data: [
                { value: counts.positive, itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: '#52c41a' }, { offset: 1, color: '#95de64' }
                ])}},
                { value: counts.negative, itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: '#ff4d4f' }, { offset: 1, color: '#ff7875' }
                ])}},
                { value: counts.neutral, itemStyle: { color: new echarts.graphic.LinearGradient(0, 0, 0, 1, [
                    { offset: 0, color: '#1890ff' }, { offset: 1, color: '#40a9ff' }
                ])}}
            ],
            label: { show: true, position: 'top', fontSize: 14, fontWeight: 'bold' }
        }]
    };
    barChart.setOption(option);
}

// ==================== 关键词 ====================
function updateKeywords(data) {
    const positiveEl = document.getElementById('positive-keywords');
    if (data.positive_keywords && data.positive_keywords.length > 0) {
        positiveEl.innerHTML = data.positive_keywords.map(kw =>
            `<span class="keyword-tag positive">${kw.word}</span>`
        ).join('');
    } else {
        positiveEl.innerHTML = '<span class="keyword-placeholder">暂无正面关键词</span>';
    }

    const negativeEl = document.getElementById('negative-keywords');
    if (data.negative_keywords && data.negative_keywords.length > 0) {
        negativeEl.innerHTML = data.negative_keywords.map(kw =>
            `<span class="keyword-tag negative">${kw.word}</span>`
        ).join('');
    } else {
        negativeEl.innerHTML = '<span class="keyword-placeholder">暂无负面关键词</span>';
    }
}

// ==================== 预览表格（含反馈列） ====================
function updatePreviewTable(results) {
    const tbody = document.querySelector('#preview-table tbody');
    if (!tbody) return;

    tbody.innerHTML = results.map((item, index) => {
        const sentimentClass = item.sentiment === 'positive' ? 'positive' :
                              item.sentiment === 'negative' ? 'negative' : 'neutral';
        const emoji = getSentimentEmoji(item.sentiment);
        const labelText = getSentimentText(item.sentiment);
        const confidencePercent = (item.confidence * 100).toFixed(1);

        // 构建反馈控件（空容器，稍后通过事件委托绑定）
        return `
            <tr data-row-index="${index}" data-text="${item.text.replace(/"/g, '&quot;')}"
                data-predicted="${item.sentiment}" data-confidence="${item.confidence}">
                <td>${index + 1}</td>
                <td style="max-width: 400px; overflow: hidden; text-overflow: ellipsis; white-space: nowrap;">${item.text}</td>
                <td><span class="sentiment-badge ${sentimentClass}">${emoji} ${labelText}</span></td>
                <td>${confidencePercent}%</td>
                <td class="feedback-cell">
                    <div class="feedback-controls">
                        <select class="feedback-select">
                            <option value="positive">正面 😊</option>
                            <option value="negative">负面 😠</option>
                            <option value="neutral">中性 😐</option>
                        </select>
                        <button class="btn btn-sm btn-primary feedback-submit-btn">提交</button>
                        <span class="feedback-msg"></span>
                    </div>
                </td>
            </tr>
        `;
    }).join('');

    // 为所有反馈提交按钮绑定事件（事件委托更高效，这里直接绑定）
    tbody.querySelectorAll('.feedback-submit-btn').forEach(btn => {
        btn.addEventListener('click', function(e) {
            e.stopPropagation();
            const row = this.closest('tr');
            if (!row) return;
            const text = row.dataset.text || '';
            const predictedLabel = row.dataset.predicted || 'neutral';
            const confidence = parseFloat(row.dataset.confidence) || 0;
            const select = row.querySelector('.feedback-select');
            const correctedLabel = select ? select.value : 'neutral';
            const msgEl = row.querySelector('.feedback-msg');

            // 禁用按钮防止重复提交
            this.disabled = true;
            this.textContent = '提交中...';

            submitFeedback(text, predictedLabel, confidence, correctedLabel, function(success, data) {
                if (success) {
                    msgEl.textContent = '✅ 已修正，感谢！';
                    msgEl.style.color = '#10b981';
                    row.style.backgroundColor = '#f0fdf4';
                } else {
                    msgEl.textContent = '❌ ' + (data.message || '提交失败，请重试');
                    msgEl.style.color = '#ef4444';
                    const btn = row.querySelector('.feedback-submit-btn');
                    btn.disabled = false;
                    btn.textContent = '提交';
                }
                setTimeout(() => { msgEl.textContent = ''; }, 5000);
            });
        });
    });
}

// ==================== 单句分析 ====================
function initSingleAnalysis() {
    const analyzeBtn = document.getElementById('single-analyze-btn');
    const clearBtn = document.getElementById('single-clear-btn');

    analyzeBtn.addEventListener('click', doSingleAnalysis);
    clearBtn.addEventListener('click', clearSingleResult);

    document.getElementById('single-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && e.ctrlKey) {
            doSingleAnalysis();
        }
    });
}

function doSingleAnalysis() {
    const input = document.getElementById('single-input');
    const text = input.value.trim();

    if (!text) {
        alert('请输入要分析的文本');
        return;
    }

    const resultEl = document.getElementById('single-result');
    resultEl.style.display = 'block';

    fetch('/api/single_sentiment', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text: text })
    })
    .then(response => response.json())
    .then(data => {
        if (data.success) {
            singleResultData = data.data;
            showSingleResult(data.data);
        } else {
            alert('分析失败：' + (data.message || '未知错误'));
        }
    })
    .catch(error => {
        console.error('单句分析错误:', error);
        alert('分析失败：' + error.message);
    });
}

function showSingleResult(data) {
    const result = data.sentiment;
    const labelText = getSentimentText(result.label);
    const emoji = getSentimentEmoji(result.label);

    document.getElementById('sentiment-emoji').textContent = emoji;
    document.getElementById('sentiment-label').textContent = labelText;
    document.getElementById('confidence-value').textContent = (result.confidence * 100).toFixed(2) + '%';

    const probs = result.probabilities || {};
    const positiveProb = (probs.positive || 0) * 100;
    const negativeProb = (probs.negative || 0) * 100;
    const neutralProb = (probs.neutral || 0) * 100;

    setTimeout(() => {
        document.getElementById('prob-positive').style.width = positiveProb + '%';
        document.getElementById('prob-positive-value').textContent = positiveProb.toFixed(1) + '%';
        document.getElementById('prob-negative').style.width = negativeProb + '%';
        document.getElementById('prob-negative-value').textContent = negativeProb.toFixed(1) + '%';
        document.getElementById('prob-neutral').style.width = neutralProb + '%';
        document.getElementById('prob-neutral-value').textContent = neutralProb.toFixed(1) + '%';
    }, 100);

    const keywordsEl = document.getElementById('single-keywords');
    if (data.keywords && data.keywords.length > 0) {
        keywordsEl.innerHTML = data.keywords.map(kw =>
            `<span class="keyword-tag">${kw}</span>`
        ).join('');
    } else {
        keywordsEl.innerHTML = '<span class="keyword-placeholder">暂无关键词</span>';
    }

    // 激活单句反馈按钮（存储当前结果供反馈使用）
    const feedbackBtn = document.getElementById('single-feedback-btn');
    if (feedbackBtn) {
        feedbackBtn.disabled = false;
        feedbackBtn.textContent = '提交修正';
    }

    document.getElementById('single-result').scrollIntoView({ behavior: 'smooth' });
}

function clearSingleResult() {
    document.getElementById('single-input').value = '';
    document.getElementById('single-result').style.display = 'none';
    singleResultData = null;
    document.getElementById('prob-positive').style.width = '0%';
    document.getElementById('prob-negative').style.width = '0%';
    document.getElementById('prob-neutral').style.width = '0%';
}

// ==================== 单句反馈事件绑定（全局） ====================
function initFeedbackGlobals() {
    const feedbackBtn = document.getElementById('single-feedback-btn');
    if (!feedbackBtn) return;

    // 避免重复绑定
    if (feedbackBtn.dataset.bound === 'true') return;
    feedbackBtn.dataset.bound = 'true';

    const select = document.getElementById('single-correct-select');
    const msgEl = document.getElementById('single-feedback-msg');

    feedbackBtn.addEventListener('click', function() {
        // 获取当前分析结果
        const input = document.getElementById('single-input');
        const text = input ? input.value.trim() : '';

        if (!text || !singleResultData) {
            msgEl.textContent = '❌ 请先进行分析';
            msgEl.style.color = '#ef4444';
            return;
        }

        const predictedLabel = singleResultData.sentiment.label || 'neutral';
        const confidence = singleResultData.sentiment.confidence || 0;
        const correctedLabel = select ? select.value : 'neutral';

        this.disabled = true;
        this.textContent = '提交中...';

        submitFeedback(text, predictedLabel, confidence, correctedLabel, function(success, data) {
            if (success) {
                msgEl.textContent = '✅ 已修正，感谢！';
                msgEl.style.color = '#10b981';
                // 可以改变背景色等
            } else {
                msgEl.textContent = '❌ ' + (data.message || '提交失败，请重试');
                msgEl.style.color = '#ef4444';
                const btn = document.getElementById('single-feedback-btn');
                btn.disabled = false;
                btn.textContent = '提交修正';
            }
            setTimeout(() => { msgEl.textContent = ''; }, 5000);
        });
    });
}

// ==================== 智能客服 ====================
function initChat() {
    const sendBtn = document.getElementById('chat-send-btn');
    const input = document.getElementById('chat-input');

    sendBtn.addEventListener('click', sendMessage);
    input.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
}

function sendMessage() {
    const input = document.getElementById('chat-input');
    const text = input.value.trim();
    if (!text) return;

    addMessage(text, 'user');
    input.value = '';

    const loadingId = addLoadingMessage();

    fetch('/api/qa_chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ question: text })
    })
    .then(response => response.json())
    .then(data => {
        removeMessage(loadingId);
        if (data.success) {
            const answer = data.data.answer;
            const msgEl = addMessage(answer, 'bot');
            if (data.data.sentiment?.label === 'negative' && data.data.is_comforted) {
                msgEl.querySelector('.message-content').insertAdjacentHTML('afterbegin',
                    '<p style="color:#ff4d4f; font-size:0.85rem; margin-bottom:8px;">💬 已识别到您的负面情绪，已为您优先安抚</p>'
                );
            }
        } else {
            addMessage('抱歉，我遇到了一些问题，请稍后再试~', 'bot');
        }
    })
    .catch(error => {
        removeMessage(loadingId);
        console.error('智能客服错误:', error);
        addMessage('抱歉，网络连接失败，请检查网络后重试~', 'bot');
    });
}

function addMessage(text, type) {
    const container = document.getElementById('chat-messages');
    const div = document.createElement('div');
    div.className = `message ${type}`;
    const avatar = type === 'bot' ? '🤖' : '😊';
    const paragraphs = text.split('\n').filter(p => p.trim());
    const contentHtml = paragraphs.map(p => `<p>${p}</p>`).join('');
    div.innerHTML = `<div class="message-avatar">${avatar}</div><div class="message-content">${contentHtml}</div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return div;
}

function addLoadingMessage() {
    const container = document.getElementById('chat-messages');
    const id = 'loading-message-' + Date.now();
    const div = document.createElement('div');
    div.className = 'message bot';
    div.id = id;
    div.innerHTML = `<div class="message-avatar">🤖</div><div class="message-content"><p><span class="loading-dots">思考中...</span></p></div>`;
    container.appendChild(div);
    container.scrollTop = container.scrollHeight;
    return id;
}

function removeMessage(id) {
    const el = document.getElementById(id);
    if (el) el.remove();
}

// ==================== 工具 ====================
function formatNumber(num) {
    return num.toLocaleString();
}

function checkHealth() {
    fetch('/api/health')
        .then(response => response.json())
        .then(data => console.log('系统状态:', data))
        .catch(error => console.warn('健康检查失败:', error));
}

window.addEventListener('load', checkHealth);