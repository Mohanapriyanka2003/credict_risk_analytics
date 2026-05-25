/* ==========================================================================
   Interactive JS Application: CreditShield AI
   ========================================================================== */

document.addEventListener("DOMContentLoaded", () => {
    
    // ----------------------------------------------------------------------
    // 1. Single Page Application Navigation
    // ----------------------------------------------------------------------
    const navLinks = document.querySelectorAll(".nav-link");
    const sections = document.querySelectorAll(".section-card");
    
    navLinks.forEach(link => {
        link.addEventListener("click", (e) => {
            e.preventDefault();
            
            // Remove active classes
            navLinks.forEach(nl => nl.classList.remove("active"));
            sections.forEach(sec => sec.classList.remove("active"));
            
            // Add active to clicked link
            link.classList.add("active");
            
            // Add active to targeted section
            const targetId = link.getAttribute("href").substring(1);
            const targetSec = document.getElementById(targetId);
            if (targetSec) {
                targetSec.classList.add("active");
            }
        });
    });

    // ----------------------------------------------------------------------
    // 2. Global Variables for Charts
    // ----------------------------------------------------------------------
    let purposeChart = null;
    let riskChart = null;
    let stressChart = null;

    // Helper to format currency
    const formatDM = (value) => {
        return new Intl.NumberFormat('en-US', {
            style: 'currency',
            currency: 'EUR',
            maximumFractionDigits: 0
        }).format(value).replace('€', 'DM ');
    };

    // Helper to format percentages
    const formatPct = (value) => {
        return (value * 100).toFixed(2) + "%";
    };

    // ----------------------------------------------------------------------
    // 3. Initialize Portfolio Stats
    // ----------------------------------------------------------------------
    const loadPortfolioData = async () => {
        try {
            const response = await fetch("/portfolio");
            if (!response.ok) throw new Error("Portfolio fetch failed.");
            const data = await response.json();
            
            // 1. Update Metric Cards
            document.getElementById("val_portfolio_exposure").textContent = formatDM(data.summary.total_exposure);
            document.getElementById("val_portfolio_ecl").textContent = formatDM(data.ecl.baseline_total_ecl);
            
            const eclRatio = data.ecl.baseline_total_ecl / data.summary.total_exposure;
            document.getElementById("val_ecl_pct").textContent = `ECL/Exposure Ratio: ${(eclRatio * 100).toFixed(2)}%`;
            
            document.getElementById("val_portfolio_pd").textContent = (data.ecl.average_pd * 100).toFixed(2) + "%";
            document.getElementById("val_portfolio_lgd").textContent = (data.ecl.average_lgd * 100).toFixed(2) + "%";
            
            // 2. Render Chart 1: Exposure by Credit Purpose
            renderPurposeChart(data.purpose_distribution);
            
            // 3. Render Chart 2: Risk Distribution
            renderRiskChart(data.risk_distribution);
            
            // 4. Load Baseline for Stress Chart
            updateStressTest(1.0, 0.0);
            
            // 5. Run initial Fairness Audit
            updateFairnessAudit(0.35);

        } catch (error) {
            console.error("Error loading portfolio metrics:", error);
        }
    };

    const renderPurposeChart = (distData) => {
        const ctx = document.getElementById("chart_purpose").getContext("2d");
        
        // Destroy old if exists
        if (purposeChart) purposeChart.destroy();
        
        // Sort and slice top categories
        const labels = distData.map(item => item.purpose);
        const values = distData.map(item => item.credit_amount);
        
        purposeChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: labels,
                datasets: [{
                    label: 'Exposure (DM)',
                    data: values,
                    backgroundColor: 'rgba(14, 165, 233, 0.45)',
                    borderColor: 'rgba(14, 165, 233, 1)',
                    borderWidth: 2,
                    borderRadius: 6,
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                indexAxis: 'y',
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => `Exposure: ${formatDM(context.raw)}`
                        }
                    }
                },
                scales: {
                    x: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8', font: { family: 'Inter' } }
                    },
                    y: {
                        grid: { display: false },
                        ticks: { color: '#f8fafc', font: { family: 'Outfit', weight: 500 } }
                    }
                }
            }
        });
    };

    const renderRiskChart = (distData) => {
        const ctx = document.getElementById("chart_risk").getContext("2d");
        
        if (riskChart) riskChart.destroy();
        
        const labels = Object.keys(distData);
        const values = Object.values(distData);
        
        riskChart = new Chart(ctx, {
            type: 'doughnut',
            data: {
                labels: labels,
                datasets: [{
                    data: values,
                    backgroundColor: [
                        'rgba(16, 185, 129, 0.65)',  // Low - Emerald
                        'rgba(14, 165, 233, 0.65)',  // Med-Low - Sky
                        'rgba(245, 158, 11, 0.65)',  // Med-High - Amber
                        'rgba(239, 68, 68, 0.65)'    // High - Sunset
                    ],
                    borderColor: [
                        '#10b981', '#0ea5e9', '#f59e0b', '#ef4444'
                    ],
                    borderWidth: 2
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: {
                        position: 'bottom',
                        labels: { color: '#f8fafc', font: { family: 'Outfit', size: 12 } }
                    }
                },
                cutout: '65%'
            }
        });
    };

    // Refresh Button Listener
    document.getElementById("btn_refresh_portfolio").addEventListener("click", () => {
        loadPortfolioData();
    });

    // ----------------------------------------------------------------------
    // 4. Loan Underwriting & Individual Prediction
    // ----------------------------------------------------------------------
    const btnEvaluate = document.getElementById("btn_run_underwriting");
    
    btnEvaluate.addEventListener("click", async () => {
        // Collect form data
        const loanAmount = parseFloat(document.getElementById("inp_credit_amount").value);
        const duration = parseInt(document.getElementById("inp_duration_months").value);
        const propType = document.getElementById("inp_property").value;
        const checkingStatus = document.getElementById("inp_status_checking").value;
        const savingsStatus = document.getElementById("inp_savings").value;
        const housingStatus = document.getElementById("inp_housing").value;
        const sexInput = document.getElementById("inp_sex").value;
        const ageInput = parseInt(document.getElementById("inp_age").value);
        
        // Derive LGD/EAD specific helper metrics (simulating realistic banking formulas)
        let propertyRatio = 0.0;
        if (propType === 'real estate') propertyRatio = 1.20;
        else if (propType === 'building society savings agreement/life insurance') propertyRatio = 0.80;
        else if (propType === 'car or other') propertyRatio = 0.50;
        
        const collateralVal = loanAmount * propertyRatio;
        const currentBal = loanAmount * 0.92; // simulated outstanding
        const monthlyPay = (loanAmount * (1 + 0.08 * (duration / 12))) / duration;

        const payload = {
            status_checking: checkingStatus,
            credit_history: document.getElementById("inp_credit_history").value,
            purpose: document.getElementById("inp_purpose").value,
            savings: savingsStatus,
            employment: document.getElementById("inp_employment").value,
            personal_status: document.getElementById("inp_personal_status").value,
            property: propType,
            other_installments: document.getElementById("inp_other_installments").value,
            housing: housingStatus,
            job: document.getElementById("inp_job").value,
            duration_months: duration,
            credit_amount: loanAmount,
            installment_rate: parseInt(document.getElementById("inp_installment_rate").value),
            residence_since: parseInt(document.getElementById("inp_residence_since").value),
            age: ageInput,
            existing_credits: parseInt(document.getElementById("inp_existing_credits").value),
            people_liable: 1,
            current_balance: currentBal,
            collateral_value: collateralVal,
            monthly_payment: monthlyPay,
            sex: sexInput
        };

        try {
            btnEvaluate.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i> Processing...';
            const response = await fetch("/predict", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload)
            });
            
            btnEvaluate.innerHTML = '<i class="fa-solid fa-wand-magic-sparkles"></i> Evaluate Application';
            if (!response.ok) throw new Error("Underwriting API failed.");
            
            const result = await response.json();
            
            // 1. Underwriting Decision Card
            const decCard = document.getElementById("val_decision_card");
            const decIcon = document.getElementById("val_decision_icon");
            const decText = document.getElementById("val_decision_text");
            
            decCard.classList.remove("approved", "rejected");
            
            if (result.is_approved) {
                decCard.classList.add("approved");
                decIcon.innerHTML = '<i class="fa-solid fa-circle-check"></i>';
                decText.textContent = "LOAN APPROVED (Low Risk profile)";
            } else {
                decCard.classList.add("rejected");
                decIcon.innerHTML = '<i class="fa-solid fa-circle-xmark"></i>';
                decText.textContent = "LOAN REJECTED (High Risk profile)";
            }
            
            // 2. Update Risk Dial
            document.getElementById("val_pd_dial").textContent = (result.pd * 100).toFixed(1) + "%";
            
            // Set dial ring color based on risk color
            const dialCircle = document.querySelector(".pd-dial-circle");
            dialCircle.style.borderColor = result.risk_color;
            
            // 3. Update LGD Recoveries
            document.getElementById("val_widget_lgd").textContent = (result.lgd * 100).toFixed(1) + "%";
            document.getElementById("fill_widget_lgd").style.width = (result.lgd * 100) + "%";
            
            // EAD and ECL
            document.getElementById("val_widget_ead").textContent = formatDM(result.ead);
            
            const eclEl = document.getElementById("val_widget_ecl");
            eclEl.textContent = formatDM(result.ecl);
            eclEl.style.textShadow = `0 0 15px ${result.risk_color}`;
            
            // Risk Tier Text
            document.getElementById("val_risk_tier").textContent = result.risk_rating;
            document.getElementById("val_risk_tier").style.color = result.risk_color;
            
            // Pricing Recommendations (Optimal credit rate premium)
            // Premium Rate formula: PD * LGD * 100 + 4.5% baseline margin
            const premiumPremium = (result.pd * result.lgd * 100 + 4.5).toFixed(2);
            document.getElementById("val_interest_rate_premium").textContent = `${premiumPremium}% APR`;
            document.getElementById("val_capital_reserves").textContent = formatDM(result.ecl * 1.5); // capital buffer 150% ECL
            
            // Fairness audit display
            let auditText = "PASS - Low Risk Group";
            if (sexInput === 'Female' && result.pd > 0.35 && result.pd <= 0.41) {
                auditText = "ADJUSTED COMPLIANCE PASS [Bias Mitigated]";
            } else if (!result.is_approved) {
                auditText = "REJECTED - Standard Risk Threshold";
            }
            document.getElementById("val_fairness_baseline").textContent = auditText;
            document.getElementById("val_fairness_baseline").style.color = result.is_approved ? 'var(--success)' : 'var(--danger)';

        } catch (error) {
            console.error("Error evaluating underwriting:", error);
            btnEvaluate.innerHTML = '<i class="fa-solid fa-triangle-exclamation"></i> Error evaluating';
        }
    });

    // ----------------------------------------------------------------------
    // 5. Stress Testing Simulation Lab
    // ----------------------------------------------------------------------
    const pdSlider = document.getElementById("inp_slider_pd_shock");
    const lgdSlider = document.getElementById("inp_slider_lgd_shock");
    
    const pdValText = document.getElementById("val_slider_pd_shock");
    const lgdValText = document.getElementById("val_slider_lgd_shock");
    
    const updateStressTest = async (pdShock, collateralHaircut) => {
        try {
            const response = await fetch(`/stress_test?pd_shock=${pdShock}&collateral_haircut=${collateralHaircut}`);
            if (!response.ok) throw new Error("Stress testing API failed.");
            
            const data = await response.json();
            
            // Update UI Stressed numbers
            document.getElementById("val_stressed_total_ecl").textContent = formatDM(data.stressed_ecl);
            document.getElementById("val_stressed_ecl_increase").textContent = `+${data.ecl_increase_pct.toFixed(1)}%`;
            
            // Render Stressed Comparison Chart
            renderStressChart(data.baseline_ecl, data.stressed_ecl);

        } catch (error) {
            console.error("Error running stress test:", error);
        }
    };

    const renderStressChart = (baseECL, stressedECL) => {
        const ctx = document.getElementById("chart_stress_comparison").getContext("2d");
        
        if (stressChart) stressChart.destroy();
        
        stressChart = new Chart(ctx, {
            type: 'bar',
            data: {
                labels: ['Baseline Expected Loss', 'Stressed Expected Loss'],
                datasets: [{
                    data: [baseECL, stressedECL],
                    backgroundColor: [
                        'rgba(99, 102, 241, 0.55)',  // Baseline Indigo
                        'rgba(239, 68, 68, 0.65)'    // Stressed Crimson
                    ],
                    borderColor: [
                        '#6366f1', '#ef4444'
                    ],
                    borderWidth: 2,
                    borderRadius: 8
                }]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                plugins: {
                    legend: { display: false },
                    tooltip: {
                        callbacks: {
                            label: (context) => `Total ECL: ${formatDM(context.raw)}`
                        }
                    }
                },
                scales: {
                    x: { grid: { display: false }, ticks: { color: '#f8fafc' } },
                    y: {
                        grid: { color: 'rgba(255, 255, 255, 0.05)' },
                        ticks: { color: '#94a3b8', font: { family: 'Inter' } }
                    }
                }
            }
        });
    };

    // Sliders event listeners
    pdSlider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        pdValText.textContent = val.toFixed(1) + "x" + (val === 1.0 ? " (Baseline)" : "");
        updateStressTest(val, parseFloat(lgdSlider.value) / 100);
    });

    lgdSlider.addEventListener("input", (e) => {
        const val = parseInt(e.target.value);
        lgdValText.textContent = val + "%" + (val === 0 ? " (Baseline)" : "");
        updateStressTest(parseFloat(pdSlider.value), val / 100);
    });

    // Preset Buttons
    document.getElementById("btn_preset_baseline").addEventListener("click", (e) => {
        togglePresetActive(e.target);
        pdSlider.value = "1.0";
        lgdSlider.value = "0";
        pdValText.textContent = "1.0x (Baseline)";
        lgdValText.textContent = "0% (Baseline)";
        updateStressTest(1.0, 0.0);
    });

    document.getElementById("btn_preset_mild").addEventListener("click", (e) => {
        togglePresetActive(e.target);
        pdSlider.value = "1.2";
        lgdSlider.value = "15";
        pdValText.textContent = "1.2x";
        lgdValText.textContent = "15%";
        updateStressTest(1.2, 0.15);
    });

    document.getElementById("btn_preset_severe").addEventListener("click", (e) => {
        togglePresetActive(e.target);
        pdSlider.value = "1.5";
        lgdSlider.value = "30";
        pdValText.textContent = "1.5x";
        lgdValText.textContent = "30%";
        updateStressTest(1.5, 0.30);
    });

    const togglePresetActive = (activeBtn) => {
        document.querySelectorAll(".btn-preset").forEach(btn => btn.classList.remove("active"));
        activeBtn.classList.add("active");
    };

    // ----------------------------------------------------------------------
    // 6. Regulatory Fairness Auditor & Bias Mitigation
    // ----------------------------------------------------------------------
    const femaleSlider = document.getElementById("inp_female_threshold");
    const femaleValText = document.getElementById("val_female_threshold_slider");

    const updateFairnessAudit = async (femaleThreshold) => {
        try {
            const response = await fetch(`/fairness?threshold=0.35&female_threshold=${femaleThreshold}`);
            if (!response.ok) throw new Error("Fairness audit API failed.");
            const data = await response.json();

            // 1. Update Gender Audit Stats
            document.getElementById("val_male_approval_rate").textContent = (data.sex_audit.privileged_rate * 100).toFixed(1) + "%";
            document.getElementById("val_female_approval_rate").textContent = (data.sex_audit.unprivileged_rate * 100).toFixed(1) + "%";
            
            const sexDi = data.sex_audit.disparate_impact;
            document.getElementById("val_sex_di_ratio").textContent = sexDi.toFixed(4);
            
            // Adjust visual indicators
            positionScaleIndicator("indicator_sex_di", sexDi);
            
            const sexBadge = document.getElementById("val_sex_compliance_badge");
            sexBadge.classList.remove("pass", "warn");
            if (data.sex_audit.status === "COMPLIANT") {
                sexBadge.classList.add("pass");
                sexBadge.textContent = "COMPLIANT [PASS]";
            } else {
                sexBadge.classList.add("warn");
                sexBadge.textContent = "NON-COMPLIANT [BIAS]";
            }

            // 2. Update Age Audit Stats
            document.getElementById("val_mature_approval_rate").textContent = (data.age_audit.privileged_rate * 100).toFixed(1) + "%";
            document.getElementById("val_young_approval_rate").textContent = (data.age_audit.unprivileged_rate * 100).toFixed(1) + "%";
            
            const ageDi = data.age_audit.disparate_impact;
            document.getElementById("val_age_di_ratio").textContent = ageDi.toFixed(4);
            
            positionScaleIndicator("indicator_age_di", ageDi);
            
            const ageBadge = document.getElementById("val_age_compliance_badge");
            ageBadge.classList.remove("pass", "warn");
            if (data.age_audit.status === "COMPLIANT") {
                ageBadge.classList.add("pass");
                ageBadge.textContent = "COMPLIANT [PASS]";
            } else {
                ageBadge.classList.add("warn");
                ageBadge.textContent = "NON-COMPLIANT [BIAS]";
            }

            // 3. Dynamic Bias Remedy Text
            const remedyText = document.getElementById("val_bias_remedy_text");
            if (data.sex_audit.status === "COMPLIANT") {
                remedyText.style.background = "rgba(16, 185, 129, 0.05)";
                remedyText.style.borderColor = "rgba(16, 185, 129, 0.2)";
                remedyText.style.color = "var(--success)";
                remedyText.innerHTML = `<i class="fa-solid fa-circle-check"></i> <strong>Compliance Achieved!</strong> Legally balanced approvals (Disparate Impact ratio: ${sexDi.toFixed(3)}) are successfully restored via Dual-Threshold mitigation.`;
            } else {
                remedyText.style.background = "rgba(239, 68, 68, 0.05)";
                remedyText.style.borderColor = "rgba(239, 68, 68, 0.2)";
                remedyText.style.color = "var(--danger)";
                remedyText.innerHTML = `<i class="fa-solid fa-triangle-exclamation"></i> <strong>Historical Gender Bias Detected.</strong> Disparate Impact ratio of ${sexDi.toFixed(3)} fails the 80% legal standard. Drag the mitigation slider to equalise group opportunities.`;
            }

        } catch (error) {
            console.error("Error auditing fairness:", error);
        }
    };

    const positionScaleIndicator = (indicatorId, diRatio) => {
        // Map 0.0 - 1.0 (or higher) to 0% - 100% slider track width
        // Regulatory boundary is at 80% mark
        const indicator = document.getElementById(indicatorId);
        let leftPercent = diRatio * 100; // e.g. 0.8 -> 80%
        leftPercent = Math.min(Math.max(leftPercent, 5), 95); // clamp slightly inside track edges
        indicator.style.left = leftPercent + "%";
    };

    femaleSlider.addEventListener("input", (e) => {
        const val = parseFloat(e.target.value);
        femaleValText.textContent = `PD ≤ ${val.toFixed(2)}`;
        updateFairnessAudit(val);
    });

    // ----------------------------------------------------------------------
    // 7. Initial App Boot Trigger
    // ----------------------------------------------------------------------
    loadPortfolioData();
});
