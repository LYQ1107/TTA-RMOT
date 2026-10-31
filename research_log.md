# TTA-RMOT Research Log (inherited historical entries preserved below)

## 2026-10-30 D-FIT 独立诊断开始

- 从965e71fe新建research/v2dance-diag-fit-20261030，保护全部PSR成果。先完成只读核对：4个768维PCD块、head2,365,440参数，原LoRA scaling1/rank/最后两层Q,V，原每步1目标window/clip1。
- 提示预期POS0.5与实际不符：657/374用p≥0.95。按提示以代码为准沿用0.95，auto NEG为有效p<0.95，低置信soft不丢弃。
- 用户重新明确共享授权：显存足够、进程少优先，最多4卡，不终止别人。先必要兼容/梯度/前5步回归、三臂200步smoke；表A先推送再允许全425推理，最终只评last-step。

## 2026-10-30 D-FIT smoke通过与实现错误保留

- 真实alpha1/8/32以及全4层Q/K/V/O rank16+head的零初始化输出均逐位一致；旧/新默认真实前5步source/KD loss最大差0。32相关CPU回归通过。
- 遇到并保留：legacy源码自哈希白名单路径遗漏；JSON5e-05被YAML1.1当字符串的head_lr；相对输出在vendor chdir后失效。只修实际问题，未改数值超参，未开放GT。
- 早期集成错误让D2/D3遇FP16梯度溢出即退出，删除了原有AMP自适应缩放。撤销这些尝试的“不可训练”分类，保留全部原目录和日志；恢复原有最多8次缩放减半/原批次重放，无优化/无跳样，按代码bug从头重跑受影响臂。
- 三臂有效200步均通过：D1保留原完成，D2/D3均原lr2e-4、无正式LR回退；D3实际2758656参数、head更新，base版本不变，GT打开0，子集SHA369cabeb…c30b6相同。
- 下一步严格CAP重载、已有模型同8500拟合、三个8973步；表A核验GitHub之前不启动完整Val推理。此前无效尝试不隐藏，也不作为教师矛盾或容量证据。

## 2026-10-31 D-FIT：共同标签基线封存发布

- Frozen/旧Naive17/旧PSR17的共同Naive8500教师拟合完成，POS门仍为0.95、656个POS；分别选中55/40/54个，NEG selected rate为0.063870/0.004335/0.008287。GT打开0，不是GT精度或新RMOT结果。
- PSR自身8500的657个POS拟合单列（47/657），不与共同标签656的分母混用；公共baselines/completion.json保存源checkpoint SHA、源验证CE、300窗口子集SHA与独立调用成本。
- 三个8973步训练继续原命令，未更改代码、超参、checkpoint选择或启动Val；本次只同步已完成的baseline小型记录。完整表A仍必须等待全部新臂结束后先发布。
- 修正最终报告构建的实际遗漏：替换占位表时原本会截掉后面的兼容/AMP撤销历史。现保留这些原始说明并标明是历史阶段状态，补一个定向回归；tests/test_diag_fit.py共16项通过（5.35s）。仅报告代码变化，训练/预测数值完全不变。
- D2原lr2e-4完成8973更新，覆盖17源视频/425目标表达/8500判断，GT打开0/冻结版本不变。全量POS617/656=0.940549、NEG1774/7844=0.226160、pair agreement0.716418；因NEG>0.10，按双条件仍不合格，不改阈值或其它配置。final8973 SHA9e149179…cfab15，source-best7750只记录。D1/D3原队列继续；本次只封存完成D2的小记录，没有完整表A或新Val推理。
- D1随后原lr2e-4完成8973更新，同8500/17源/425目标、GT0/冻结不变。POS522/656=0.795732、NEG436/7844=0.055584、pair agreement0.761194；严格80%需525个POS，差3个仍判不合格，不四舍五入或选中途模型。final SHA b58568b3…b96d49，source-best2750只记录。D3继续原配置，表A门与全部新Val推理仍未开启。
- 补齐公开源码映射：准备执行快照a67f4ada与GitHub2259228b具有相同tree98831e0d；对当前6dc3d52比较flex_tta/flex_uda/compat/三配置/训练入口，tracked diff为空。仅静态代码复现核验，不访问GT或改动运行进程；结果将随完整表A发布。
- 三个完整训练全0退出/8973更新/原lr2e-4/无回退/同300子集SHA/目标GT0。CAP head确实更新、冻结版本不变，2758656参数；POS641/656=0.977134、NEG995/7844=0.126849，仍因NEG>0.10不合格。source-best8250仅记录，唯一待评final8973 SHA a95b57fc…020214。
- CPU汇总完整8500的表A与300窗口曲线，并实际检查PNG；三臂均不合格，不用曲线子集或loss代替判定。表A SHA d6cb3b5b0092f9579bb8743209676b9a2655d062cf3f188512022809c4f7c61f；先推送并核验，再打开完整预测阶段，GT仍不进入训练/拟合。
- 表A已实际推送并核验GitHub4a812e3359307c975f5c5a889d789c0a734e161d/tree2829d697，创建含表A SHA的发布门。启动推理前把原通用driver“four method lanes”的固定描述改为实际lane数量，只有元数据修正、没有任何数值或候选改动。三模型仍需全部425预测封存后独立GT评测，不因拟合未过而省略臂。
- 推理运行中定位离线集成问题：prepare_psr_controls原会为CLI成本更正覆写run_metadata，破坏D-FIT整体封存SHA。新增显式保留模式，把修正成本写入inference_costs旁文件、默认旧行为不变；D-FIT离线入口传此模式并在GT前复核原封存，成本报告读取旁文件。只改离线成本/记录，不改变健康推理的模型、候选或数值。
- 交付打包补齐原本未覆盖的推理stdout/矩阵资源准入与总wall time/官方TrackEval stdout、完成回执和小型推理provenance；完整预测仍只提供路径SHA。CLI成本保留封存回归通过，最新定向文件17项通过（4.55s），不把CPU锁等待算GPU推理，也不把单方法wall time当包含排队的整个矩阵wall time。
- 三模型25视频/425表达完整预测全部0退出并封存；官方9个新评测全部0退出，离线归因/排序/视频CI/表B和REPORT均已生成。最后package步骤因DIAG_FIT_*通配符也匹配派生count-controls、缺method字段而KeyError；仅修为明确注册三方法枚举并补回归，18项通过（4.33s）；仅恢复失败打包，不重跑任何GPU/官方评测/密集归因，保留原失败stdout。

## 2026-10-31 — D-FIT完整验收与停止

- 三臂8973更新、25Val/425表达完整预测和官方评测已完成。HOTA Frozen8.2180，D1 29.7020（+21.4840），D2 22.8440（+14.6260），CAP27.7680（+19.5500）；25视频聚类CI均正，但不是盲测/Train→Val。
- Frozen取各模型逐帧k后29.4000/22.8540/27.1030，反向新模型取Frozen计数7.8978/7.8380/8.1589。主要整体收益可由数量/存在性解释；该反事实依赖适应模型k，不能描述为免费独立系统，没有GT调阈值。
- CAP无教师标签帧R@1=.237412相对Frozen.156615有额外排序信号，D1/D2 .160755/.161810提升较小。Empty FP156758/686166/446071均高于87371，原始IDSW也增加，不宣称身份或空查询问题解决。
- 严格80%/10%拟合标准三臂均False，保留用户规则类别而不把它冒充已证实的输入/标签矛盾原因。所有目标GT仅封存后离线使用；评测后再验原预测、margin、run_metadata与final checkpoint全部SHA匹配。
- 最后打包KeyError已在1b24a579/publicf17ec79修复，18项回归通过，只恢复package且exit0；9官方/密集归因/GPU实验未重跑，原失败stdout保留。训练尝试15.3862GPU-process-h、完整推理2.5681GPU-worker-h，推理矩阵6199.17s/实际GPU1和4，继承/验证成本另列不冒充免费。
- 本轮全部实验已停止，只做最终GitHub记录交付。未启动任何新增密集GT标注、列表蒸馏、源端长训或第七种TTA；后续研究等待用户下一条指令。
- 最终交付补齐六个count-matched对照的完整官方summary/detailed CSV，明确其父模型成本；只复制已有结果、不重算指标，打包回归同时检查其不被记为独立GPU方法。最新同一18项回归通过（4.66s），不与历史选择相加。

## 2026-10-30 — Three-seed main official confirmation, not best-seed selection

- Seed42all8officialTrackEvalexit0; full425Naive7.8064/PSR8.8763,Frozen8.2180. Together17/0/42,Naive7.6174±.17921984,PSR8.7222667±.26144258(samplesd). AllthreePSRpointvaluesaboveFrozen,meanDelta+.5042667; meanvsNaive+1.1048667. No fakeFrozenindependentreplicates.
- Countdiagnosisconsistent:PSR@Frozencount8.0336/8.0312/8.0207allbelow8.2180. Frozen-onlycountPSR−Naive.9330/1.2341/1.0167; pointbenefitnotautomaticindependentsemanticgain. U/MTcontrolsremainseed17only,notthree-seedsuperiorityproof.
- Seed0fullanalysis/tables/exportcomplete, seed42analysisrunningoriginal25053. Parentofficialandfinalfullsummarypending. Addedcorrect-positive-fitforseed0fromitssealedofflineCSV; correctPOSselectedNaive39/363,PSR41/374. No training/threshold/model/sourcecheckpoint changes.
- ImmediateGitHubrefverificationoncewasstaleafterAPIwrite; freshref+commit+exacttreeconfirmed9d484bae. No experimentalrerunsorforcewrites; publishernowre-readsrefbrieflyaftercheckingwriteerrors.

## 2026-10-30 — GPU closure and seed0 complete official result

- Finalparentfull425inferenceexit0:2975.33475s, original56047closed0; allownGPUtraining/inferenceendednormally, no newGPUwork. OriginalCPU25053beganseed0controls+8officialTrackEvalall0, nowdenseofflineanalysis; remainingseed42+parentCPUphasesqueued.
- Actualall13receipt/fullpredictionauditPASS:each8973updates/24576params/17source/25target/425queries/3588445candidate-queryrows/433636activequeryframes,source-selectedcheckpointSHAequalsinferenceSHA,FP32threshold0/testupdates0/GTattempts0/boxes+IDsunchanged. Savedall13_students_and_full_predictions.json; do not confuse this with officialall47completion.
- Seed0full425official:Frozen8.2180,Naive7.5959,PSR8.8701. Froz@Naivecount7.4002,Froz@PSRcount8.6343;PSR@Frozen8.0312. PSR−Naive1.2742,butFrozen-onlycountdifference1.2341; no semantic/noveltysuccessclaimorbest-seedselection. Seed42/parentmetricsstillpendingoriginalqueue.

## 2026-10-30 — Four confirmations sealed; GT-free count controls prefetched

- Original84940registeredseedpairsparentexited0; all4newseed0/42studentsfull25/425predictionssealed. Onlyparentcontrolinferenceleft, latest20/25 onoriginal56047; originalCPU25053stillwaitsparentbeforeanynewGT.
- Existingprepare_psr_controls --precompute-only ranonceforeachcompletedseedroot, CPU0newinference/teacher/GTOpens;8directionalcontrols×425episodes×25videosPASS. Bothcontrols_sealed.json remainabsent,GTauthorizationfalse. No newmethods/training queued.
- SavedinitialCPUcompute times/score-and-countparentsinconfirmations/control_prefill_stage.json before finalprepare reuse; totalderivedcoststillincludesparents. Thehelperlegacyremaining-list namesprimary8, notextra confirmationjobs; actualregisteredmethodsareonlyNaive/PSRin eachseedroot.

## 2026-10-30 — All required students trained; parent reweighting actual acceptance

- All13newstudents(8primary+4registeredconfirmation+1parentcontrol)completed8973updates. Parenttrainexit0 elapsed2629.20s/body2623.41s,17source/25target/425queries/8500teacherlabels/24576params,checkpoint roundtrip/frozenunchanged/GTattempts0,adapterDeltaL2=7.2102811.
- Parentcontrolactualnewweaklabels0,originalendpointreplayterms526,POSrepeatedexposures2240over815/8973updates. Weightedreplay/directgradcosmean.9378790; atstep8000all4directPOSrawBCEandparentBCEexactly2.0473876, extraweightedgrad=.25directgrad, demonstratingreweightingnotnewlabels. No HOTA inferredfromthis.
- Threeof4confirmationfull425inferencesexited0, seed42PSRlatest15/25; parentfullinferPID3860174onGPU6since1793301285.951654,latest5/25. No furthertrainingqueued; originalCPU25053waitsremainingcompletepredictionsbeforeGT.
- Snapshottedall5additionalcompletedtrainingsandvalidatedremainingexecution/publicruntimepackages:seed42both0b5bd8,parent035a153. Fullcorepackageshashesmatch; no livecode/modelprocesschanges.
- Read-onlyreviewforward_uda+soft_distillation confirmsrawlastdecoder-layerlogitdifference,independentcandidateBCE,nodouble-sigmoidorhistory-axiserror. InitialFP32equivalenceselectionmismatches0. Observedpoorcorrect-positivefitispartialtransfer, notproofthatabackbone/optimizerbugexists; no protocolretuning.

## 2026-10-30 — Correct-positive transfer, not only aggregate loss

- Reusedsealedprimary4policyCSVandexistingsoft_fit_by_state, rawGTfiles/GPUcalls0; addedanalyze_psr_correct_positive_fit.py. No model/config/budgetchanges.
- CorrectteacherPOSofficialmatch:Naive363/source38/student37(new26/lost27);PSR374/source30/student38(new30/lost22);Uniform371/source19/student25;Confidence370/source39/student44. PSRGeometry38/Discovery36, NaiveTemporal39. Countsarecandidateanchors, notuniquepersonsorindependentvalidation.
- PSRcorrectPOSsoftBCE8.5279126→2.0038101/meanprob.0921734→.2209508, yet89.84%stillnotselected. Partialsupervisiontransferexists butdecisiongatecoveragepoor; no source-only/targetGTthresholdorcapacityretuning. Linkreport inhumanfailure/verdict/results.
- RecheckedlatestuserPSR5.1–6.4requirements:allmain+Uniform/Confidence/Discovery/Geometry/Temporal/matchedMTimplemented; paired3seedconfirmationrequiredandactive. DeterministicFrozenreference reusedexplicitly, notthreefakeindependentFrozenruns. ParentPOSreweightingaddressesrequiredexpansionconfoundwithoutnewlabels.

## 2026-10-30 — Final report augmentation prepared without experiment changes

- Originalwait248confirmedlive. seed0NaiveandPSRfull425inferexit0; seed42bothtrained8973exit0andstillinfer. ParentweightcontrolGPU6progress4000/8973, sameconfig/sourceGTonly; no newGPUjob orGTmetricpeeking.
- Addedreport-onlyfinalize_psr_confirmation_delivery.py, gatedonoriginalfullCPUcompletion. It refreshesold-importparentreplaydisplay, addsfullstudentcost/separatevalidationcost, copiesactualinterface/smoke receipts andouterCPU/schedulerlogs. No model/seed/threshold/checkpoint changes.
- Realregression: all8primarytraining/inferencecostfields exactlyequalpreexistingofficialsummary, matchedMTexternalteacherbudget0. No parentreplaytermscountedasnewweaklabels; inheritedteacher/Frozen/frontcostsseparate.
- Fixedactualgenericexportrecoverypath bug: seedroot/teacher_full doesnotcontainthe sharedteacher. Resolvefromimmutabletrainingconfig+pseudoseal.teacher_receipt; existingseed0NAIVE/PSRpaths matchprimaryandexist. FutureCPUexportswillusethecorrectpaths; primaryhistoricalmetricsunchanged.
- Clarifiedpubliccodehistoryreproduction: serverhasprivateexecutioncommitlineage; freshclonecanverifypublicmatching_snapshot+SHA. Sourcecodeauditsalreadyprovefullruntimepackageequality, notjustfilenamesorintent.

## 2026-10-30 — Actual execution source and completed confirmation training published

- Fetchedonlycurrentresearchbranchpublichistory, noHEADswitch orhealthyprocessinterruption. All8primaryrecordedcodeSHA256 matchactualexecutioncommitsANDfetchedpublicsnapshots; entireflex_uda/flex_tta/compat/source-loader packages havezerodiff. Naive/PSRcorepublic733f00d; remainingcore0b5bd8. Addedpublication_map/source_revisionaudit, notfalseGitcommitidentityequivalence.
- All3newcompletedtrainings(seed0Naive/PSR,seed42Naive) also audited againstpublic0b5bd8 core; completion8973/24576/17source/425target/GT0. Snapshotfulltrainingtraces/receipts/sourceval/stdout/checkpointSHAintoindependentconfirmationpaths, noGPUreruns/targetGT/earlyevaluationseal.
- Originalblockingorchestrator248continues; lateststreamseed0Naive15/25,seed0PSR10/25,seed42Naiveinferactive,seed42PSR>=8000/8973. ParentweightcontrolCPUqueueunmodified. CurrentgoalturnclassificationPROGRESSwithactualsource+artifactauditsandpublictrainingrecords, notunexecutedplan.

## 2026-10-30 — Delivery/reproducibility audit while confirmations run

- CPU-only openedall26publishedpredictiontarballs andofficialCSVseach:425predictions/25videos/425officialepisodes; all8finalstudentreceipts8973/24576/17source/425target/GTattempts0. Actualvendor trackedpatch matchesalreadypublicvendor_tracked_source.patch; no livevendor/code/modelprocess modified.
- Captured two actualenvironmentversionmanifests viaimportlib metadata, noGPUallocation orpipinstall. AddedPSR_REPRODUCIBILITY commands/assetpaths/sourceonlydataaliases/patchapplication/recoveryprotocol; oldLocateMOTcode/weights remainoutofscope. Addedaudit_psr_delivery.py forrealartifactchecks, notscientificsuccess.
- Originalnative waitorchestrator248 still owns84940confirmationpairs/56047parentqueue/25053CPUclosure; no duplicatejobs. seed0bothtrainedexit0;425inferencecontinues; seed42Naive>=8000/PSR>=6000 lateststream. Continuefixedbudget/params, noGTthresholdchanges.
- Deliveryregression exactselected10CPUtestfiles:75passed5.16s; compileallflex_uda pass. This is the current relevant scope, not a sum with earlier79or16overlappingtests.
- Freshformalassetdeliveryaudit:40Train41796images/1560queriesstillcomplete,0/40samefrontendMOT/nofrontendprovenance; noGTpayloadread orfrontend substitution. Savednewformal_train_audit_delivery.json, originalauditpreserved.
- Weakoriginbreakdown fromsealedofflineJSON:PSR167bothendpoint+middlecorrect,3oneendpoint+middlecorrect,84zeroendpoint+middlewrong. No two-correct-endpoints/middlewrong cases; observedweak errors inherit teacherunmatchedpositives, not proven newtemporalidentityfailure. Query-visibleGTlimitretained.
- Report-onlybranchrename distinguishesparentreplayfromnewweaklabels; primaryPSRtrainingdynamicsexactJSON equalityPASS. RunningCPUclosure importedprevioussummaryfunction, so refreshparent dynamics afteritscompletionbeforefinalpublication. No training/evaluationnumericschanged.

## 2026-10-30 — First complete PSR matrix, registered confirmation and weight confound

- All8 students8973updates and25/425 predictions sealed, original/repair GPU parents exited; official26methods exit0. PrimaryHOTA Frozen8.2180, pairedNaive7.4499, PSR8.4204, Uniform8.4347, Confidence6.7173, Discovery7.8127, Geometry8.4218, Temporal7.9524, matchedMT8.6095, Source7.4352. No independent PSR novelty/success claim: Uniform and MT higher; Full belowGeometry.
- Registered conditionPSR>Frozen+Naive triggers pairedstudentseeds0/42 with immutable sampling17/caches,8973updates/24576params/lr/selection unchanged. Four lanesGPU6/5/4/7; no foreign tasks stopped. Independent CPU count/ranking/coverage analysis remains in original blocking parent.
- User5.2 requires geometry benefit not merelypositive-weight/sample-count gain. Added explicitlyPOST_HOC PSR_PARENT_WEIGHT_CONTROL: exactlysame263geometry opportunities, but extra0.25normalizedmeanBCE uses526 actual endpoint-parent terms at their original frames/probabilities, no new weak labels/frames/teacher calls. Same8500direct cache andfixedstudentbudget. Queue after a confirmation lane really exits; first actualinterface+6step smoke, thenfull. Not part of preregistered headline, no GT enters prediction/training, nooriginalthresholdchanged.
- Dedicated newmethod dispatch only; existing8method loss paths unchanged. Parent replay counters explicitly distinct from uniqueweak evidence; control is opportunity/weight matched, not an exact matched-gradient intervention.

## 2026-10-30 — Complete primary official/count/ranking/coverage closure

- Original blockingCPU pipeline completed controls40.13s, official26methods186.61s, offlineanalysis836.36s, tables0.08s, export30.98s; GPUreruns0.26complete425prediction archives+officialdetailCSVs+logs+all8trainingtraces exported.
- PSR−Naive+0.9705, butFrozen@PSRcount−Frozen@Naivecount+0.9330; reversePSR@Frozen8.0336<8.2180. Do not claim count-independent semantic recovery.
- OfficialPresent40/Empty385 unchanged; PresentHOTA9.7456→8.8043, EmptyFP87371→20059. FrameIoU05 recovery8773/lost8025/net748, newFP18327/removal85724; matchingreassignment kept distinct.
- PSR374correctdirectPOS (344source-rejected), vsNaive363 (net11); weak170/254correct,84errors. DirectGTcoverage0.176%,newweak0.080%; Uniform371directcorrect andbetterHOTA.
- FullcandidateAUROC.650204→.756113/AP≈.1618, butGTone-to-oneR@1/5notbetter andqueryswapadvantagedeltaCIcrosses0. AssA21.147→16.245/IDSW269→338. PSR−FrozenvideoCI[-6.3931,5.9808],14improve/8decline/3unchanged. ScientificindependentcontributionNOTestablished.
- Regression16PSR+MLLM tests pass after adding dedicatedparentcontroldispatch. Registeredseedconfirmation remainsrunning, no thresholds/lr/capacity/teacherbudget changed.
- Added officialCSV-onlyleave-one-video sensitivity: PSR−Frozenremaining24range[-1.38465,3.76537],4negative; PSR−Naiveall25positive[.79595,1.01488]; PSR−Uniform21negative. No models/GTthresholdsrerun. APIlarge537entrytreewrite timedout afterallblobs uploaded; recovered64entrytreebatches, exacttreeverified, remote c396caf5. No result edits or experimentreruns.

## 2026-10-29 — New PSR implementation goal; user explicitly reauthorizes25Val

- Start861840e, clean repository, independent research/v2dance-psr-trainval-20261029. Rechecked40Train1560queries/41796images:0/40matchingMOT, officialZIP has no prediction payload. OfficialiKUN issue35/44 author confirms NeuralSORT/NKF code not released. No detector replacement orGT substitution.
- User explicitly says still no40 and useVal, and reauthorizes memory-fit/few-process GPUsharing,max4. This stage is sameVal adaptation, NOT formalTrain→Val or blindtest; old8.0327 preserved, not reused as new paired-samplingNaive.
- Implemented actual five-role budget discovery, bounded positive-only bridge expansion and separate normalized losses. All17DanceVal querytypes include action/mixed semantics; only one weak middle frame between two full-expression teacherPOS endpoints, no unbounded action/ID propagation. Extra appearance/overlap checks onlyFull; source-calibrated0.95/0.05 kept.
- Preregistered paired5anchors×2tracks×2singleframes=20calls/query, same425queries and8500calls in each ofNaive/Uniform/Confidence/PSR. All four real immutableMOT/Frozen-score sampling manifests builtGT-free;6targeted tests pass. Source commit recorded861840e during uncommitted new sampler construction; the new code is saved in the immediately following PSR stage commit, not falsely attributed to oldP1 source.
- Main students retain24576params/seed17/8973updates/lr5e-5/source-val-onlyselection. Naive/PSR first, then required ordinary sampling/propagation controls. No claim of success before real training and full425official evaluation.

## 2026-10-29 — Complete first naive frozen-MLLM distillation,25Val diagnosis

- User explicitly authorizes25Val first because40Train official MOT is absent; not Train→Val, unseen-video generalization or onlineTTA. P1/oldUDA/sixTTA bytes preserved.
- Actual seed17 student8973steps complete; all17source videos/8973windows and25target videos/425queries/8381valid soft teacherpairs covered,35348related KD exposures. Source-val-only selectedstep8973, frozenparametersunchanged, targetGTopens0, no MLLM/testupdates during inference.
- Four whole-video workers on GPUs1/0/3/4 completed25/425, officialTrackEval five methods exit0. Frozen8.2180, Source7.4352, MLLM8.0327; dualcount Frozen@MLLM7.9355 andMLLM@Frozen8.0166. No HOTA improvement overFrozen; do not start Ours or extendtraining.
- CandidateAUROC.650204→.754863/AP.093446→.160565, conditionalGT-presentAUROC.539649→.660466, and nonanchor gains exist. One-to-oneGTRecall@1/3/5 is not improved consistently. Loss decrease is not semantic success; teacher-positive fit and count/ranking/association are key failure modes.
- OfflineIoU05 recovered7797/lost7696GTobservations; newFProws16147, removed85655,7existingrows becomeFP byrematching. Counts differ from officialtemporalCLEAR, not hidden changes to metrics.
- Train1544.73s/.42909GPUh, fourworker inference3357.92s/.93276GPUh with1073.20s schedulerwall. Teacher cache reuse0newcalls but inherits8381real P1 calls/483.78amortizedforwardseconds, not a zero-cost teacher system.
- Complete failure attribution, all predictions/receipts/logs and actual cost provenance before proposing one-variable nextdiagnostic. NoDanceGT threshold/prompt/checkpoint tuning, no automatic E4–E7 or second-domain expansion.
- FinalCPU text controls reused the established evaluator: correct-vs-visual mixed-frameAUROC advantage delta+.001731,25videoCI[-.020807,+.034617]; correct-querytop1.614788→.613076. Presence mean-marginAUROC.660260→.754026 is an offline full-episode statistic, not a causal presence classifier.
- Fixed-anchorGT1102 decomposes exactly into97noMOT/523recoverable-unsampled/66sampled-but-teacher-belowhalf/416teacher-abovehalf-covered. Officialempty≠exhaustive semanticabsence; Qwenpretraining overlapunknown. Capacity不足notproved; only next64pair/200step GT-free single-capacity diagnostic registered, not executed.
- Delivery preserves five850-member archives(425predictions+425originalreceipts each),68training/smoke logmembers and39evaluation logmembers,2125candidate-scorefile SHAs/paths. Reference receipt packaging issue fixed without prediction rerun; first regression harness failure retained separately and corrected to use its temporary package root.

## 2026-10-26 — User-authorized25Val transductive pilot: real smoke and Frozen gate

- User explicitly chooses25Val first, not Train. Keep original40Train asset limitation separate; label all new results
  TRANSDUCTIVE_VAL_DIAGNOSTIC, never independent target generalization or source-free onlineTTA.
- Initial AMP BCE failure fixed by FP32 presence loss; independent complete loss review then found NaN source totals
  from frozen C-Hook cross-query dispersion denominator N(N-1) at N=1. Revoke ALLv1 smoke approvals despite finite
  gradients; retain old runs. Exclude only this inapplicable frozen constant, keep original decode and CE unchanged.
- Freshv2 four real20-step runs have finite actual source/target losses, nonzero adapter updates and target gradients,
  no frozen source changes, no targetGT access. Final receipt indentation failure repaired and resumed20/20 without
  replaying optimizer steps. UDA focused tests23 passed. Independent audit rather than exit0 is the acceptance gate.
- Frozen CPU replay425/425 exact prediction hashes, complete211085MOT bank: official HOTA8.2180 unchanged;
  DetA3.1937/AssA21.1470/IDF14.4414. Historical8.206 remains separate. Replay cost not standalone GPU inference.
- Proceed with pre-registered one8973-step coverage cycle for A/B/C/D, source-val CE-only checkpoint selection,
  then full25/425 sealed official evaluation. No target-presence truth is available: UNKNOWN coverage is explicit,
  no fabricated MIL positives, no40/385 prior, no test-time optimizer. No target score-based hyperparameter changes.

### Full-cycle AMP numerical recovery

- Source first fails at step2666 after2500 checkpoint. A real offending source window at the latest valid state has
  finite loss~2.47, Inf gradients at AMPscale131072, finite at65536/32768 and FP32. This is scaled FP16 overflow,
  not targetGT-driven diagnosis or a reason to skip a difficult sample. Teacher/Weak also hit the same numerical guard.
- Resume only failed workers; healthy Combined keeps its already loaded configuration. Exact-sample backoff restores
  RNG, memory, evidence and counters, halves loss scale, and retries without optimizer/EMA/sample advancement.
  Preserve and archive all suffix traces past the last checkpoint before replaying that unsaved suffix.
- Separate repair admission/status files avoid clobbering the still-active original four-worker launcher.
  Loss weights/seed/lr/threshold and source-val-only model selection remain fixed;26 focused tests pass.

### All four training runs complete; first complete official controls

- All4 accept8973 updates each,35892 total, no skipped samples,3 same-sample AMP retries each, fixed source17/8896
  query coverage and target25/425 for B/C/D. Every accepted objective/gradient finite; frozen base unchanged; targetGT model opens0.
  These are sampled-window cycles: source393 unique anchors, target75 unique anchors/1275query windows, not all-frame optimization.
- Source and Teacher each seal all425 predictions before any GT evaluation. All four models are already trained and
  source-selected before these early official evaluations; remaining inference is read-only and no hyperparameter changes follow.
- Official Source Rehearsal HOTA7.4352 (−0.7828),AssA28.1110 but DetRe2.3022 vs Frozen4.5441. Source CE improves
  while cross-domain referring recall falls: do not call pruning-driven association averages an identity or semantic solution.
- Mean Teacher HOTA8.2494 (+0.0314),CLRTP10780/FP89488/100268 outputs; its Count-Matched Frozen8.2479 has the
  same CLEAR TP/FP/output totals. Only+0.0015 remains relative to that dependent diagnostic. Source Count-Matched7.4207.
  This is not a causal percentage or evidence of a robust semantic gain. Continue Weak/Combined full inference unchanged.
- C/D also add source-presence loss, so their difference from A is not a pure target-constraint ablation. Target MIL/presence
  positive/negative coverage0, allUNKNOWN; D temporal coverage0 for mixed dancing+clothing queries. State these limitations explicitly.

## 2026-10-25 — Identity full result and automatic Hybrid handoff

- Identity completes25 videos/425 expressions,3,588,445 candidate-query rows/433,636 active frames. Official HOTA8.8179 (+0.5999),DetA4.0132,AssA19.3760,DetRe5.9354,DetPr11.0030,IDF1 5.4476,IDSW347;114,648 rows/1,968 IDs. All25 workers exit0, full auditPASS with no source-sign or MOT modifications and no GT opened by adaptation.
- Net GT recovery+3,223 (3,247 recovered/24 lost), FP net+11,721; newly selected unmatched12,445/removed735 remain distinct. Empty-query FP97,792 (Frozen87,371), nonempty rate42.5974% (Frozen40%); present recall6.3799% (Frozen4.8635%). AUROC0.650204→0.642678 and allTopK/Recall@K decline. This is recall/output expansion, not demonstrated semantic or identity-consistency improvement; AssA falls and IDSW rises.
- Video-cluster bootstrap:17 up/2 down/6 ties;95% CI[-0.083812,+2.430322] includes0. Removing0007 leaves+0.161939. Retain the larger positive point estimate without a stable-gain claim.
- Actual cost8940.97s wall/9.8481GPUh,12.4047 workerFPS/48.4999 schedulerFPS, peak18,289,683,968 allocatedbytes.108,069 finite nonzero-gradient attempts,86,678 accepted/21,391 norm-limit rollbacks, maxL2 1.99999856. All425 pre-episode resets pass, all25 frozen-gradient flags false and rollback-restored flags true. Full-scope post-reset smoke flags are absent, not presumed true. No hyperparameter/threshold change.
- Hybrid starts automatically onGPU0/4/6/1, initialPIDs581747/581814/581900/581985, same strictV2 weights/completeMOT/defaults. Do not interrupt it or start a seventh full run. Six-method closure and later presence-aware mechanism/prototype remain outstanding.
- Additional offline cost attribution reads only sealed audit/GT-group summaries: Identity Present/Empty accepted updates8,630/78,048, rollbacks2,352/19,039; Rank10,805/95,166 accepted. Validate group sums against full-run counters and label Guard reuse. Rule-eligible history is not semantic correctness; the Empty group is90.04% of accepted Identity updates yet gains10,421 FP. With episode resets, these updates cannot benefit other Present episodes. This motivates investigating unlabeled presence evidence, never GT-gated online adaptation.
- Group-counter regression and full suite:51 tests passed in11.89s, compileall passed; all five real grouped-counter checks and exact official HOTA pooling pass with unchanged bootstrap comparisons. No online module, hyperparameter or completed prediction is changed by this reporting extension.
- Prepared a static all-six-only scientific plotting entry point for paired video intervals and25-video heterogeneity. It reads sealed numerical summaries only and refuses interim data; render/visual validation remains pending until Hybrid completes. This is a report artifact, not a new model or GPU experiment; the presence-aware prototype remains deferred.
- Final cost wording also identifies the shared frozen-score preparation: table stages are measured conditional on that asset. Rank/Hybrid need the frozen-source stream; standalone online deployment must include its computation, even if frozen encoders can be shared. No unmeasured teacher+student end-to-end FPS claim is made.
- Found inherited LocateMOT project routing in this repository's README/local AGENTS despite verified TTA-RMOT origin/branch and actual FlexHook runs. Archive the old README verbatim with an explicit historical header; route this repository to TTA-RMOT and link its actual research artifacts. Keep generic/long-running rules unchanged. No LocateMOT/GroundJEV worktree, process, model or dataset is modified, and healthy Hybrid continues unchanged.

## 2026-10-25 — Supplemental scientific acceptance, no online changes

- User supplement adds Query-Present/Empty grouping, 25-video paired uncertainty, identity-memory risk and Hybrid-vs-single-module checks. Keep healthy Identity and queued Hybrid unchanged; no seventh full run, source training, GT tuning or teacher.
- One requested live check confirmed all four GPU0/4/6/1 worker processes,43 distinct sealed episodes and healthy completion logs. Thereafter use the existing blocking full-suite session, not repeated process/file polling.
- Added isolated `offline_query_diagnostics.json` beside (not replacing) all four old attribution files; all preexisting global recovery/FP/AUROC/TopK results assert unchanged. Empty episode nonempty rate:Frozen40%, Relative100%, Rank/Guard40.2597%. Present recall:4.8635%→Rank4.9365%; Empty FP87,371→88,057. Keep all425 episodes.
- Paired bootstrap uses25 whole video clusters/2000 draws/seed0 and exact official HOTA pooling. Rank11 up/8 down/6 ties,95% CI[-0.008807,+0.142421]. Removing0007 leaves only+0.002743 HOTA gain. No robust semantic/significance claim; no GT feedback to running jobs.
- Separate real multi-target candidate Recall@K from prior any-hit TopK. Added active-frame quartile negative-output diagnostics and explicitly limited query-GT-visible identity-assignment conflict checks; the latter are not true full-MOT ID error rates or proof of accepted bad updates.
- Final report now requires all six query diagnostics and all-six video-cluster results, reports evidence coverage/parameters, the identity-recall-loss classification and Hybrid comparisons. Next-stage presence-aware design/prototype is deferred until the six-method closure; do not invent independent semantics if only source self-consistency is available.
- Verification:50 focused tests passed in12.55s; compileall/shell syntax passed. All four diagnostic backfills verified their preexisting global statistics against original published JSON exactly, without touching completed GPU outputs. Rank sensitivity: removing0007 leaves+0.002743, removing0065 gives+0.060659; these are nonadditive leave-one-video-out diagnostics, not contribution percentages.
- Tightened final scientific wording: a positive HOTA/net-recovery point estimate with a video-cluster interval including0 is explicitly insufficient evidence of a stable gain, not a research-success verdict. This changes reporting only, not any current online process, parameter or prediction.

## 2026-10-25 — Full Rank/Guard: tiny HOTA gain, not a semantic breakthrough

- Published all four completed methods, official per-expression/per-video outputs and Rank/Guard analysis (GitHub8353006). Exported actual experiment-commit→identical public-tree mappings for all four, including Guard derivation and preserved vendor tracked patch. Added compact official evaluator exit-status receipts to each published method. Relative CPU cost, like derived Guard, is explicitly incremental and depends on prior GPU source scores. No online algorithm, seed, sample or prediction changes.
- Full 25-video/425-expression Rank: HOTA8.2433 (+0.0253 points versus matched8.2180), IDF1 4.4947, AssA21.0150, DetRe4.6120; net GT +155, diagnostic FP +723, AUROC0.650204→0.648328. Genuine 105,971 finite/nonzero gradient updates, zero rollbacks. Keep the small positive result and its negative ranking evidence; do not claim robust semantic improvement.
- Full Guard: HOTA8.2444 (+0.0264), IDF1 4.4956; retains31 Frozen-positive rows over Rank, +158 GT/no lost baseline GT, FP +751. Direct real GPU full-episode equivalence passed:1,203 frames/5,911 selected rows, byte-identical predictions,282 updates, no GT,112.43s. Full CPU derivation7.34s reuses Rank's5.1463 GPU hours; incremental CPU cost is not standalone RMOT cost.
- Reporting bug: additive episode deltas zeroed structural constants. Correct only published summary fields from actual shard stats (24,576 trainable parameters, PCD768 per replica); preserve raw sealed metadata, predictions, numerical defaults and counters. Label Guard's reused updates/cost dependency explicitly rather than implying no adaptation happened.
- Rank wall5,541.59s; source-only static group estimates left a backward-compute tail. Identity is now running a queue of25 complete-video jobs (17 independent episodes each), max4 registered GPUs0/4/6/1; Hybrid follows. No episode memory cuts, no GT-driven scheduling, no re-run of finished Rank. All six remain required before goal completion.
- Verification after the reporting-only correction:47 focused tests passed in11.97s; compileall and shell syntax passed. All four published methods retain425 valid episodes/25 videos/3,588,445 candidates with no changed official scores. Guard's complete real-episode equivalence receipt is published alongside the metrics.

## 2026-10-25 — User-authorized GPU sharing and pending-only repartition

### Completed full controls and first live gradient method

- ZERO_UPDATE: all 25 videos / 425 expressions, 3,588,445 candidate-query observations / 433,636 active frames. Official HOTA/DetA/AssA/IDF1 = 8.2180/3.1937/21.1470/4.4414; DetRe/DetPr 4.5441/9.6868, IDSW269, 99,704 rows / 1,769 IDs. This is the new matched baseline, not TTA gain over historical 8.206.
- Six historical sign differences were all probed: cached vs uncached original selects the same complete candidate IDs at every case; full Frozen CPU replay has zero prediction-row differences. Preserve the historical audit, native traces and the matched-reference receipt separately.
- RELATIVE_ONLY full: HOTA7.5705 (delta -0.6475), DetA4.1857, AssA13.7120, IDF1 5.5746, DetRe11.1040, DetPr6.2855, IDSW829, 375,483 rows / 8,014 IDs. Recovered14,873 GT, total FP +260,906; source AUROC/TopK unchanged. Do not claim semantic improvement or overall success.
- Candidate-retrievable GT:203,906/212,541 (95.9373%); Frozen gate matches10,337 (5.0695% of retrievable). Source Top1 61.4788% vs random57.8299% on target-retrievable frames. Weak stage-two referring selection, rather than missing boxes alone, is a key bottleneck.
- PF_RANK started on GPU6/4/0/1, PIDs4129095/4129171/4129260/4129353, whole-video capacity groups119/119/102/85. All remaining methods will still run; first-control failure is not goal completion.
- Additional sealed-output diagnosis: 385/425 episodes have no scored GT, 40 have GT; all remain in the primary benchmark. Official GT_Dets212,541 matches the diagnostic GT row count. Frozen empty/present-episode FP =87,371/1,996; RELATIVE_ONLY =337,014/13,259. No-match rejection is a major bottleneck; do not mistake conditional target-frame ranking for query-presence detection. Only offline reporting changes; no GT feedback to TTA settings. Framewise diagnostic assignment and CLEAR temporal assignment are explicitly distinct.

- This checkout is `/data2/user/FlexHook-TTA`, remote `LYQ1107/TTA-RMOT`; inherited LocateMOT entries below are history, not the current task.
- ZERO_UPDATE shard0 completed 6 videos / 102 expressions, exit 0. The old idle-only coordinator then waited while GPU5 was occupied by a foreign training job.
- User explicitly authorized sharing when VRAM fits, preferring four GPUs with fewest processes. Stop only our waiting coordinator PID 3682643; retain all completed predictions, original manifest and resource/status snapshots.
- Change: occupied-card eligibility requires 24 GiB worker budget + 8 GiB free headroom; prefer fewer compute processes, then more free memory. Bind UUIDs and record physical indices; no foreign process is stopped. PyTorch allocator cap is not a resource reservation.
- Repartition only 19 unstarted whole videos into four pending groups; retain completed shard0 verbatim. Episode seed, weights, query bank, online memory and update defaults do not change. No target GT is read during the transition.
- Numerical results remain pending; resource sharing may affect throughput and does not imply a 4x speedup.
- Verification: all 39 focused tests passed in 10.71 s, including occupied-card memory/process ordering and completed-shard-preserving repartition; compileall and shell syntax checks passed.

## 2026-10-25 — Near-zero legacy score differences and truly matched reference

- Interim ZERO check: 895,696 candidates, two historical sign differences, max margin discrepancy 8.58307e-4. Preserve the failed assertion and underlying raw outputs; do not turn either row into an adaptation gain.
- Actual diagnosis on complete candidate sets: video0081 frame908/ID915 and frame760/ID947 have the same source selection in uncached original and cached FP32 execution (zero sign/ID differences). Both differ from the old file only around zero. No target GT, optimizer update or threshold adjustment was used.
- User already approved rebuilding the matched Frozen baseline. Implement a common frozen reference using actual complete ZERO_UPDATE margins, preserving all old logits, supplement, original traces and separate historical comparisons. No invented logits: downstream methods read the real scalar margins at their chronological frame.
- Before opening GT or continuing TTA: diagnose every full-run historical sign difference using original/cached GPU forwards, require zero original/cached sign differences, then require exact independent CPU score-to-MOT replay on all 425 sealed episodes. This replay is not claimed as an independent second GPU model experiment.
- All six methods use the same actual-source reference for comparison and downstream historical evidence; no gain is measured against historical 8.206. 41 focused tests passed in 10.34 s, including immutable matched source streams and old receipt compatibility. Full-run diagnosis/reference sealing remain pending.
- Later review fixed an existing gate bug: initial-source checking used the global optimizer count instead of this episode's count. Enforce it before backward for every matched-reference episode; full audit rejects any initial mismatch. 42 focused tests pass in 10.04 s. This does not modify live ZERO workers or their already sealed outputs.
- Added an all-six-only final scientific report generator. Explicitly separate average worker FPS, actual scheduler-wall FPS and extra validation/evaluation cost; do not infer 4x speedup from four GPU processes. Report generation and all scientific results remain pending.
- Matched-source IO now reads only actual per-episode sealed margins instead of parsing the unused 394 MB legacy JSON in every worker; legacy hashes and supplement provenance remain checked. The matched CPU replay regression passed (43 tests, 10.11 s).
- Official evaluator source filters GT by nonzero mark and pedestrian class 1; align the offline diagnostic loader to those same GT rows instead of importing an unfiltered legacy loader. This was checked on synthetic GT only, without opening actual target annotations before full sealing. Total FP change is reported separately from new/removed selected-row FP because matching assignments can change. Final regression: 44 tests passed in 10.77 s; compileall and shell syntax passed.
- Measured completed ZERO shards: GPU6/4 ~37 active frames/s versus GPU0 ~29, with GPU1 slower under sharing. A fresh-method heterogeneous LPT plan now uses source-only timings and active-frame costs, assigning more complete videos to faster eligible GPUs. Card selection still prefers fewer processes and sufficient memory; new-card capacity is explicitly estimated. No accuracy/GT-based scheduling, no changed samples, seed or adapter state, and no rebalancing started shards. 45 tests pass in 10.06 s; compileall/shell syntax pass. Publish each method's actual manifests and launcher summary with its metrics.

Concise experimental log (latest first).

## 2026-10-20 — G2 step3000 full MOT17-02 expansion

- The preserved G2 Clean step3000 bundle and the original G0 used the same 600-frame MOT17-02-FRCNN manifest, official_scoring parser, tracking_dense decoder, Simple association and TrackEval commit. Runtime read no GT or future frames; the evaluator opened the registered GT.
- Official full-video result: G0 HOTA/DetA/AssA/IDF1 `0.357098/0.667771/0.194362/0.330325`, IDSW/FP/FN `822/1621/3212`; G2 step3000 `0.502578/0.716588/0.353559/0.495423`, `413/1664/2780`. G2 delta is HOTA `+0.145479`, IDF1 `+0.165097`, IDSW `-409`, FN `-432`, with 63/600 rejected frames versus G0 48/600.
- Judgment: this is a complete registered development diagnostic and supports a positive G2 front-end result on this video. MOT17-02 is Clean TRAIN-scope (the audit found exact frames 302–305 and 602 same-video TRAIN rows), so it is not an independent generalization claim and does not isolate JEV identity decisions. A video-disjoint development evaluation remains required.
- Receipt: `docs/GROUNDJEV/G2_FULL_MOT17_02_EVAL_20261020.{md,json}`; the protected G2 owner continued independently and reached step5447/18227.

## 2026-10-20 — V1 complete partition matched result

- Hypothesis: query probes plus a causal external referring adapter could improve referring selection while leaving TempRMOT detection, QIM, boxes, and physical IDs frozen.
- Scope: Refer-KITTI-V1 videos `0012` and `0014`, 56 video-expression episodes and 5,320 chronological frame rows. The runtime manifest had no evaluator fields; the runner reported `gt_used_by_runner=false`, `future_frames_used=false`, and `evaluation_manifest_opened=false`.
- Resource: the GPU selector chose physical GPU 2 because it had sufficient free memory and zero compute processes. The run recorded GPU UUID `GPU-c2484ded-5f87-237c-bbf0-d23fcb74fdb2`.
- Official TrackEval result: FROZEN HOTA 86.618; EMA_ST 83.899; PROBE_CACHE 85.543; PROBE_ST 86.627; QCS_TTA 86.531; NO_STATE 86.544. Deltas versus FROZEN were -2.719, -1.075, +0.009, -0.087, and -0.074 respectively.
- QCS trace: 916 optimizer steps; `L_eq` was nonzero on 880 steps and `L_excl` on 19 applicable steps (maxima 0.04486 and 0.00715). The negative result is therefore from an executed logic-loss method, not a zero-loss placeholder.
- Judgment: QCS-TTA did not improve the complete V1 partition. PROBE_ST's +0.009 is a small control fluctuation, not a method-success claim. Retain the full run and all sealed evaluator files.

## 2026-10-20 — Logic losses wired and V2 matched confirmation

- Failure found: the first controller version implemented `L_eq`/`L_excl` functions but set both terms to zero in the replay update because aligned probe probabilities were not carried into evidence records.
- Fix: CandidateEvidence now retains aligned q0/q_eq/q_cf probabilities; a bounded CPU logic cache supplies stop-gradient q_eq/q_cf targets while only adapted q0 logits receive gradients. Empty masks and missing counterfactuals remain no-op.
- Verification: compileall and 18 tests passed. In the corrected V2 0005/`automobiles-are-located-to-the-left` 297-frame trace, QCS-TTA made 75 updates and both `L_eq` and `L_excl` were nonzero on the applicable updates.
- V2 official TrackEval: FROZEN HOTA 67.591; QCS_TTA 67.509; NO_STATE 67.509. The QCS delta was -0.082, so the corrected implementation still showed no gain.
- Retain: corrected run at `results/runs/v2_0005_complete_episode_qcs_logic`; prior zero-logic run remains as a failure/fix artifact rather than being overwritten.

## 2026-10-20 — V1 launch argument failure and recovery

- Failure: an initial full V1 command passed `--episodes 0012 0014`, but the runner contract requires `video/expression` keys; it failed before model loading with a missing-episode `KeyError`.
- Fix: reran without an episode filter because the V1 runtime manifest already contains only videos 0012/0014. No prediction or evaluator output from the failed launch was reused.

## 2026-10-20 — QCS-TTA source boundary, V2 assets, and first smoke

- Hypothesis: a frozen TempRMOT pre-QIM referring snapshot can support causal, source-free external adaptation without changing detector, QIM, or physical IDs.
- Asset result: official TempRMOT source commit `6a65640d849fdee4a32bb055945ee34c3b0edeb1` and V2 `checkpoint_rk2.pth` were loaded from `/data2`; checkpoint size is `1096316824` bytes and SHA256 is `4953bd2b2966f752b7d5dced94216f0ddacd04158b1c85c3fc3534a333e9581b`. V2 test-video manifests cover 457,503 runtime/evaluation rows and runtime contains no GT fields.
- Implementation: added the frozen wrapper/snapshot, rule-only query variants, isolated probe/shadow state, IoU alignment, set-valued evidence, bounded replay and semantic state, external residual/HEAD257 adapters, causal runner, GPU selector, official TrackEval boundary, and required run/report entry points.
- Failure/fix sequence retained: direct runner import lacked project `sys.path`; source/probe score masks mixed CPU/CUDA tensors; static probe replayed excess history; each was fixed and a V2 one-frame smoke then completed with zero source checkpoint missing/unexpected keys and a successful TrackEval invocation.
- Smoke result: V2 video `0005`, expression `automobiles-are-located-to-the-left`, one frame, methods FROZEN/QCS_TTA; source parameters frozen, QCS optimizer step count 1, no GT opened by runner. TrackEval produced files, but this is not a formal accuracy result because the sequence was truncated to one frame. Retain as interface evidence only.
- Resource decision: GPU selection now filters by free memory and prefers fewer compute processes; the observed eligible choice was physical GPU 2. No other project process was stopped or modified.
- Retain: V1/V2 assets, manifests, source sanity, smoke artifacts, and all failure logs. Pending: full matched pilot, evaluator summaries, and final scientific conclusion.

## 2026-10-18 — Expanded held-out gate to all three task loops

- The held-out owner now preserves DT1203 MOT and additionally schedules the
  registered RMOT single/multi validation scopes plus TAO validation videos 4
  and 63 with official multitask metrics. It passed a dry-run at J1 step300 and
  was replaced as a CPU-only owner without changing training or GPU ownership.

## 2026-10-18 — J1 step300 and held-out gate reattached

- The live J1 three-task run reached a complete step300 checkpoint with
  adapter weights and full FP32/Adam/RNG state. MOT20-05 remains in its
  protected G0 export without a `result.json`; expression-conditioned RMOT has
  not started inference and remains gated on that release.
- The CPU-only held-out gate owner had exited while waiting, so it was
  reattached to the live J1 PID without restarting training.

## 2026-10-18 — GPU5 G1 functional retention control

- With explicit user authorization, GPU5 was shared after a memory check (about
  68.6 GiB free); the foreign TALON jobs were preserved. The corrected owner
  completed G0/G1 MOT4, RMOT single8, and RMOT multi12 controls in the OneJev
  checkout. HOTA changed 0.7485→0.7333, 0.2718→0.2117, and 0.2179→0.2284;
  this is short functional evidence, not a full-dataset promotion result.
- The G1 service was cleaned up and the evidence/documents were synchronized
  to the GroundJEV GitHub branch. The active J1/MOT20/RMOT pipeline was not
  interrupted.

## 2026-10-18 — DeepSeek Harness handoff re-audit and active continuation

- The requested GroundJEV-MOT redirection was checked against the actual
  DeepSeek Harness SSH-job records and live child processes. Harness launcher
  records with `exit_code=0` are not completion receipts: the detached
  children for the three-task SFT and MOT20 export are still alive. The live
  research checkouts remain `/data2/user/LocateMOT_OneJev_CVPR` and
  `/data2/user/LocateMOT_GroundJEV`; this root checkout is still the required
  project identity `/data2/user/LocateMOT`.
- Active work: `trackjev_3task_J1_v4_birth` (PID 3029659, ranks 3029872 and
  3029873) is a real MOT/RMOT/OVMOT run over 43,533 rows (21,168 / 16,677 /
  5,688), using GPUs 4 and 6. The latest observed log is step 280/1,200 and
  step 250 has a complete FP32-master/Adam/scheduler/RNG restore state; the
  old MOT-only training has no residual process.
- Protected export: MOT20-05 `mot20_MOT20-05_resumed_v2` (PID 3743898) is
  still restoring the exact 1,560-frame prefix and running with the explicit
  4 GiB native-feature budget after the earlier 2 GiB failure. It has no
  `result.json` yet. The expression-conditioned RMOT pilot (PID 3744099)
  remains correctly gated on that receipt and has performed zero model
  inference or optimizer updates.
- GPU5 is not idle: its active compute owner is another user's
  `/data2/OCD/TALON` process. The existing GroundAnything G1 evidence already
  contains a complete 200/200 projector adaptation and G0/G1 comparison; the
  measured result is a retained negative detector adaptation. No foreign job
  was stopped and no duplicate G1 run was started.
- Next recovery point: let the protected MOT20 export and J1 checkpoint reach
  their registered completion boundaries, then run the prepared
  expression-conditioned RMOT pilot and an independent held-out evaluation
  from the latest valid J1 adapter. No benchmark success or unified-goal
  completion is claimed in this audit.

## 2026-10-01 — latest R1 restoration completed

- Hypothesis: restore only the newest R1 execution boundary and its legal
  public-data/runtime inputs on `/data2/user/LocateMOT`; do not pretend that
  unavailable L69/L89E/R1 payloads were reproduced.
- Restored: official KITTI training images, Refer-KITTI V1, Refer-KITTI V2,
  MMDetection 3.3.0/GroundingDINO, BERT, and the direct R1 source dependency
  modules. V1 audit is complete. V2 has 157 legal expression-frame target
  references (11 repeated physical video/frame/ID pairs) pointing to IDs
  absent from corrected labels; missing frames are zero, and the anomaly is
  retained for the next adapter instead of silently filtered. The new public
  query manifest records every affected row.
- Runtime evidence: GroundingDINO loads the SHA-verified official checkpoint
  with 172,977,693 frozen parameters; one real KITTI frame/expression GPU
  inference returned 300 finite candidate boxes and finite scores. R1 imports
  and compile checks pass after path migration. The safe source loader also
  reads all 8,397 legal V1/V2 expression records from the restored
  per-expression layout without opening official-evaluation videos.
- Failure/limit: historical R1 checkpoints, L69 bank, dense indexes,
  language/visual/aligned caches and fixed manifest are unavailable. No old
  HOTA or score is claimed as reproduced; the old R1 sidecar is not extended
  with threshold/layer/learning-rate sweeps.
- Next action: construct a new legal public-data manifest/index and then a
  freshly generated candidate bank before any new correspondence training.
  The manifest is now complete. A 40-row midpoint-frame GroundingDINO probe
  found mean best IoU `.3188`, IoU@.25 `.6585`, IoU@.50 `.1463`, and mean best
  score `.0107` over 123 target boxes; the frame-zero attempt is preserved as
  a separate artifact. This supports a new bank audit but is not an RMOT
  score. Official-evaluation videos remain unread.

## 2026-08-19 Stage L11 — OVMOT temporal pseudo-track + KITTI front-end

- Hypothesis: unmatched DLA detections on expanded TAO-train OVMOT
  stream must receive high-precision temporal pseudo-track supervision
  instead of "every detection is NEW"; KITTI RMOT DetPr ~1% is a
  candidate front-end problem, fixable with category whitelist + NMS +
  query-conditioned CLIP top-k.
- Implementation:
  - Pseudo-track generator (forward + backward cycle consistency,
    appearance/motion/category filters, confidence gating) for all 500
    TAO train videos; sidecars in `outputs/l11/data/pseudo_tracks`.
  - Class-A GT upgraded from C-TAO base @0.5 IoU to base_and_novel
    @0.3 IoU (~10% -> ~28% coverage).
  - Trainer: no_unmatched_new, confidence-weighted identity/relevance
    losses, pseudo ids in the same slot/lifecycle machinery.
  - KITTI: whitelist + cross-NMS + top-30 -> ~15.3 cands/frame (was 50);
    CLIP top-12 calibration: query precision 10.2%, target recall 65%.
- Quality: pseudo same-ID precision 99.26% (8-video audit, target 90%);
  cycle pass 0.997; NEW rate 0.110 vs L10 collapse ~1.0.
- Running: 4-GPU repair training (from L9-ovmot, ~6-8 s/step),
  repaired-KITTI eval (L9-ovmot ckpt), 10-video OVMOT over-birth
  baseline: unique/rows 0.211 (L9), reused IDs 467.
- Pending: step-7000 OVMOT quick eval, full TAO TETA, KITTI TrackEval,
  L12 prompt seeding.

## 2026-08-20 continued

- Low-LR continuation (10000->11000, LR~6e-6): RMOT-Dance 32.49 ->
  34.05 HOTA; ordinary Macro 0.4855 -> 0.4924.  OVMOT full TETA on
  s11k running.
- Fresh-optimizer balance run (p_rmot 0.45/p_ovmot 0.20) REJECTED at
  step 1000 (RMOT 32.06 / ordinary 0.4663).
- L12 frozen prompt-seeded (DAVIS 2017, 10 multi-object videos):
  mask/point > box for identity robustness; joint fine-tune not
  launched (mixed signal).  Infrastructure complete.

## 2026-09-04 Stage L88 — RMOT-aware GroundingDINO LoRA adaptation

- Hypothesis: a zero-initialized rank-16/alpha-32 LoRA on the final two
  GroundingDINO fusion layers and decoder layer 0, with the frozen L86/L87
  sidecar and causal history, would improve held-out target-bag and
  expression-to-candidate correspondence without changing the L69 bank.
- Contract: label-free, distributed, and temporal regressions passed; full
  V1/V2 fit used seed `20260829`, 40 epochs, 2,640 optimizer steps, world 4,
  effective group batch 8, and 20 even checkpoints. Local MMCV required FP32
  rather than the registered BF16 autocast. Candidate deletion/truncation was
  false and all frozen-base/LoRA gradient and reload checks passed.
- Dev: all 20 checkpoints were scored on 138 video-disjoint groups; the fixed
  selection chose epoch20/Rule B before fixed validation. Checkpoint SHA is
  `7012706140cfa94278ce15bb8da3e2318eb3efcfb8d26d3b1dbc5206f1145538`.
- Fixed semantic validation: immutable L29 was
  `recall=.7333333, precision=.0830189, FP/frame=10.125,
  pred/positive=8.8333, hard=.9166667, multi=.8194444`. L88 candidate-only
  was `recall=.4193548, precision=.1830986, FP/frame=2.4167,
  pred/positive=2.2903, hard=.8461538, multi=.3333333`; final Rule B was
  `recall=.1935484, precision=.1363636, FP/frame=1.5833,
  pred/positive=1.4194, hard=.8461538, multi=.1666667`, with empty rate
  `.5416667` and inactive false acceptance `.5`. The semantic gate failed on
  recall and multi-positive preservation; the lower volume is not a fix.
- Internal TrackEval: selected Rule B produced HOTA `26.0914` (V1) and
  `20.2386` (V2), below L86 `29.1663/21.6467` and L87-A
  `28.5752/22.1300`. No screening or official-test labels were read.
- Preserved failures/fixes: first formal causal-key drift, sparse seqinfo,
  merge, device, and score-shape retries remain in new directories with their
  provenance; no old bank/checkpoint/production entrypoint was overwritten.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`; no L88 long continuation,
  screening, official test, TrackEval beyond internal validation, or
  ordinary MOT/OVMOT change is authorized. Unique next action is supervisor
  review of one new RMOT correspondence/proposal design.

## 2026-09-05 Stage L88C — corrected candidate-vs-NULL replay

- Hypothesis: the prior L88 semantic failure may include an implementation
  error in the candidate-vs-NULL gate. Corrected emission is
  `candidate_energy >= candidate_threshold` AND
  `candidate_energy-null_logit >= null_margin` AND
  `presence_logit >= presence_threshold`; candidate-only uses only the first
  condition. No training, backward, optimizer, checkpoint update, bank
  change, top-k/NMS or NULL suppression was performed.
- Implementation: refit all 20 even checkpoints on the registered B/R/P
  grids, replayed the corrected shortlist (epochs 8/20/40/30/4) over complete
  V1/V2 dev video, ran the corrected internal TrackEval matrix, froze epoch
  30 Rule B before fixed validation, and ran the fixed 16-calibration/
  24-validation replay. A first TrackEval provenance alias failure was
  preserved; the minimal `scope_key/full_video` alias fix was pushed in
  `c5a6685` and the corrected matrix is attempt2.
- Result: L88C corrected final on validation was
  `recall=.354839, precision=.207547, FP/frame=1.75,
  pred/positive=1.7097, hard=.846154, multi=.305556,
  empty=.375, inactive-FA=.666667`. Immutable L29 remains
  `.733333/.083019/10.125/8.8333/.916667/.819444` for
  recall/precision/FP/pred-positive/hard/multi. Candidate-only recall was
  `.419355`; the corrected NULL comparison removed additional positives.
- Internal dev TrackEval for the frozen epoch30/Rule B was V1
  HOTA/DetA/AssA `.296615/.216472/.408302` and V2
  `.265682/.156622/.451938`; these are dev-only, not screening or official
  results. The semantic gate failed on recall and multi-positive recall.
- Root cause: existing candidate scores have useful average pairwise ranking,
  but the lowest-scoring positives in multi-target bags do not survive the
  fixed emission boundary; V2 is worse. This is a correspondence and
  multi-positive recall failure, not a universal NULL-acceptance,
  coverage, finite/reload, or implementation-smoke failure.
- Preserved outputs: corrected replay, two TrackEval attempts, frozen
  selection, fixed semantic JSON/gate, and offline A–I diagnosis under
  `outputs/l88c/`. Code was pushed to branch
  `codex/l88c-candidate-null-corrected-replay-20260905` at `e8f5f62`.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`. No screening/official-test
  labels, HOTA claim beyond internal dev TrackEval, ordinary MOT/OVMOT/TAO
  change, or new L88C training was run. The only next action is the single
  supervisor approval request in `reports/l88c/NEXT_TEST_APPROVAL_REQUEST.md`.

## 2026-09-06 — L88C recoverable storage cleanup

- While the authorized zero-training L88C final-internal replay was running,
  its open paths were checked. Three superseded L1 intermediate LoRA
  checkpoint directories (`outputs/l1_c/checkpoints/lora/checkpoint-100/200/300`,
  about 8.0G each) were not open by any process and were moved, not deleted,
  to `/data2/usr_for_deadline/locatemot_cleanup_20260906/`.
- Frozen banks, formal reports, current L88 epoch30 checkpoint/cache, and all
  ordinary MOT/OVMOT assets were left in place. `/data1` free space increased
  from about 76G to 114G. The recoverable inventory is documented in
  `reports/l88c/L88C_RESOURCE_CLEANUP_20260906.md`.

## 2026-09-07 Stage L89 — query-conditioned set correspondence decoder

- Hypothesis: a fresh QSC-D head over the frozen L84/L87-A Z1 candidate states,
  with candidate self-attention followed by cross-attention to pure frozen
  GroundingDINO/BERT language tokens, could improve same-frame correspondence
  without changing the bank, detector, tracker, or ordinary MOT/OVMOT.
- Implementation: isolated worktree at
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89`, base
  `ece9fc5549a9210424eb127fb10430c3fed7b7ba`; two set layers, eight heads,
  FFN 1024, hidden 256, history 8, 4,169,829 trainable parameters. Pure
  language cache has 3,418 label-free entries. Contract smoke passed after
  preserving two package/field failures. No token/span or static/motion
  supervision is verified (`UNALIGNED`).
- Fit: seed `20260829`, one GPU, BF16, 40 epochs/2,640 optimizer steps,
  finite loss and nonzero gradients, all 20 even checkpoints plus final.
  Only L49 fit supervision was used; no screening or official-test labels.
- Dev/selection: 20 checkpoints scored over 138 legal dev groups; five-item
  shortlist and full-video B/R/P matrix selected epoch 4, Rule R before fixed
  validation. Checkpoint SHA is
  `5ab3cb344b73b320b34de7b4bb41622ce665ecb17c4d90c1999640a318e69aa8`.
- Fixed semantic: authoritative retry4 failed. L29 was
  `recall=.7333333, precision=.0830189, FP/frame=10.125,
  pred/positive=8.8333, hard=.9166667, multi=.8194444`; L89 was
  `recall=.8709677, precision=.0794118, FP/frame=13.0417,
  pred/positive=10.9677, hard=.6923077, multi=.9027778`, with inactive
  false acceptance `1.0`. Hard-negative ranking moved, but deployable volume,
  precision, and no-match behavior failed.
- Internal TrackEval: frozen epoch4/Rule R on V1 validation (86 sequences)
  gave HOTA `2.2906`, V2 validation (537 sequences) HOTA `0.5894`, versus
  L87-A `28.5752/22.1300`. Full-video post-hoc target-row recall was only
  `.0131/.0026` and inactive acceptance was `1.0`. This is internal
  validation TrackEval evidence, not screening or official-test evidence.
- Root cause judgment: query-conditioned set representation gave limited
  frame-level ranking signal but failed calibrated persistent emission and
  absence handling. The branch is stopped pending supervisor review; no L89
  continuation, screening, official test, or ordinary MOT/OVMOT change is
  authorized. All L89 outputs are preserved under `outputs/l89/` and the new
  source/reports are on branch `codex/l89-query-conditioned-set-correspondence-20260907`.

## 2026-09-08 Stage L89C — corrected candidate-vs-NULL replay

- Hypothesis/repair: L89 training and the registered loss use `candidate_energy`
  against `null_logit`, but the old L89 evaluation/deployment path emitted
  using `presence_logit - null_logit`. L89C changed only the evaluation,
  selection, inference, and TrackEval contract to the shared
  `corrected_emission_mask`: candidate threshold, candidate-minus-NULL margin,
  and presence threshold. No training, scorer forward, bank/cache rebuild, or
  model/ordinary-MOT change was made.
- Inputs/selection: immutable L89 raw dev records (9,960; SHA recorded in
  `outputs/l89c/dev/corrected_shortlist_attempt1/`), 20 existing checkpoints,
  and the fixed manifest SHA
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
  Corrected dev TrackEval evaluated 5 checkpoints × B/R/P (15 results), then
  froze epoch 2 / Rule R, checkpoint SHA
  `f8b175597ece8aad1f0ec3ae9d05c70e0759a0f7480dff9af6ea2619a2e3f08b`, with
  candidate/presence/NULL thresholds `-0.75/-1.0/0.0` before fixed validation.
- Fixed semantic evidence: authoritative retry2 has 40 ordered records (16
  calibration, 24 validation), with preselection forbidden-label fields
  absent. Validation is recall `.8387097`, precision `.0869565`, FP/frame
  `11.375`, predictions/positive `9.6452`, hard violation `.6923077`,
  multi-positive recall `.8333333`, inactive false acceptance `1.0`.
  The gate is `semantic_gate_fail`: hard/recall/precision/multi floors pass,
  but volume and inactive/no-match conditions fail.
- Internal evidence: frozen epoch-2/Rule-R full-video inference covered V1
  `0004,0018` (86 sequences) and V2 `0016,0017,0020` (537 sequences).
  Corrected TrackEval HOTA is V1 `2.244718%`, V2 `0.583059%`; no screening or
  official-test labels were read. These are internal validation-scope metrics,
  not a final RMOT result. Historical buggy L89 was `2.2906%/0.5894%` and
  L87-A was `28.5752%/22.1300%` on V1/V2.
- First actionable root cause: corrected candidate-vs-NULL accounting still
  leaves no-match acceptance universal and output volume too high. All rows,
  keys, finite checks, and frozen boundaries passed; the initial GPU OOM is
  preserved as an implementation/resource attempt and the exact CPU retry
  completed. Unique next action: supervisor review and authorization of one
  new RMOT branch explicitly addressing absence/volume calibration with
  candidate correspondence. Do not extend L89/L89C or tune thresholds.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`; `zero_training=true`,
  `screening_gt_used=false`, `official_test_labels_read=false`,
  `ordinary_mot_ovmot_touched=false`.

## 2026-09-08 Stage L89D — true full-video timeline repair

- Hypothesis/scope: repair only the L89C sparse-prediction versus dense-GT
  timeline mismatch. No training, model, loss, bank, tracker, threshold, or
  ordinary MOT/OVMOT change was made. New code is confined to `tools/l89d_*`
  and new evidence is under `reports/l89d/` and `outputs/l89d/`.
- Contract repair: native L69 `frame_ids` and frame pointers now drive every
  video timeline; all query/frame pairs and all candidate rows are scored.
  The multi-video query-map overwrite was fixed and covered by a targeted
  regression. Dev inference attempt3 covers 1,152,690/1,152,690 candidate
  pairs across 2,385 frame groups; internal attempt1 covers
  243,550/243,550 pairs across 1,844 frame groups. Both passed full-video and
  pair-completeness checks. The dev and internal dense Z1 supplements were
  built from frozen query-independent caches and finalized with no labels.
- Dev evidence: the valid 5-checkpoint × B/R/P TrackEval matrix selected
  epoch 8 / Rule P on the preregistered dev tuple. The frozen checkpoint is
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_L89/outputs/l89/train/joint40/checkpoint_l89_epoch008.pt`,
  SHA256 `28d4c35a189474d446a26fc096bbafbad77e1298429674383ce7ce8b14677d99`.
  The fixed 16-calibration/24-validation semantic replay then failed: recall
  `.5483871`, precision `.1517857`, FP/frame `3.9583333`,
  predictions/positive `3.6129032`, hard violation `.6923077`,
  multi-positive recall `.4861111`, inactive false acceptance `.6666667`.
  Recall and multi-positive floors failed; no screening/test labels were read.
- Internal evidence: fixed epoch 8 / Rule P true full-video TrackEval gives
  combined HOTA `23.8586078862%`, DetA `15.1294290119%`, AssA
  `37.9461647913%`, DetRe `57.2559333985%`, DetPr `16.9954368707%`, IDF1
  `18.6931722536%`, and IDSW `6337.5`; V1 HOTA `25.0798607896%`, V2 HOTA
  `22.6373549828%`. These are internal validation-scope TrackEval results,
  not screening, official-test, or final ordinary-RMOT evidence.
- Attempts preserved: dev ENOSPC, internal exit-137/OOM retries, the expected
  smoke full-video rejection, the selector directory-argument error, and the
  earlier multi-video GT-map mismatch. The final boundary guard passed after
  the path contract correction. The fixed manifest SHA remained
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
- Status: `L89D_COMPLETE_ZERO_TRAINING_TRUE_FULLVIDEO_REPAIR /
  semantic_gate_fail / STOPPED_PENDING_SUPERVISOR_REVIEW`.
  Flags: `zero_training=true`, `screening_gt_used=false`,
  `official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`.
  Unique next action: supervisor review and authorization of one separate
  absence/volume-calibration study; do not extend L89/L89C or alter the
  production MOT/OVMOT paths.

## 2026-09-09 Stage L89E — phase-consistent temporal/history replay

- Hypothesis/scope: L89D evaluated every checkpoint with temporal history
  enabled even though L89 training used Stage-S epochs 1–8 with temporal off
  and zero history. L89E changed only this replay contract: S epochs 2/4/6/8
  use `temporal_enabled=false` and zero history; T epochs 10–20 and J epochs
  22–40 use causal last-four history. No training, model, loss, bank, tracker,
  threshold science, UIDM, ordinary MOT or OVMOT source was changed.
- Implementation: isolated branch/worktree based on L89D commit
  `d0960e1d1679414765f79203bef444f0f9968928`; phase policy and wrappers are
  in `tools/l89e_*`. The required CPU equality check passed for S and T
  `L86ClipStore` history construction. The first replay failure was a
  `FrameExample`/`_clip_history` interface mismatch; the minimal adapter fix
  passed a targeted S/T causality regression. A subsequent historical-score
  merge check found that the immutable sparse file contains S+T/J rows; the
  wrapper now filters only the historical T/J epochs and preserves their
  values. Launch-path and shortlist argument mistakes were preserved as
  `INCOMPLETE`/operator evidence; no affected model run was treated as data.
- Stage-S evidence: fresh phase-correct replay of epochs 2/4/6/8 completed
  `4 x 498 = 1,992` records, all finite, complete and zero-history. Merged
  phase-consistent fit/dev pool contains 20 epochs and 9,960 records. The
  registered maximum-five fit/dev shortlist was frozen at epochs
  `8(S),20(T),40(J),32(J),4(S)`.
- Dev selection: full native dev replay retained `11,003,527/11,003,527`
  candidate rows for all five shortlist candidates. The 15-result TrackEval
  matrix selected epoch 4 / phase S / Rule B by the registered tuple; dev
  HOTA was `0.296512` (internal fit/dev selection evidence only).
- Fixed semantic evidence: the new 16-calibration/24-validation replay used
  the frozen epoch-4/S zero-history policy and Rule B. Immutable L29 remains
  recall/precision/FP-frame/pred-positive/hard/multi
  `.7333333/.0830189/10.125/8.8333/.9166667/.8194444`. L89E validation is
  recall/precision/FP-frame/pred-positive/hard/multi
  `.4516129/.1458333/3.4166667/3.0967742/.7692308/.3611111`, with inactive
  false acceptance `.8333333` and empty rate `.2083333`. The semantic gate
  fails recall and multi-positive floors despite lower volume and higher
  precision; no threshold rescue was performed.
- Internal TrackEval: after the frozen selection, exact internal timeline
  replay covered `243,550/243,550` pairs for V1 0004/0018 and V2 0016/0017/0020.
  Phase-consistent HOTA/DetA/AssA/DetRe/DetPr were V1
  `28.7628/18.7680/44.3927/80.3045/19.5312%` and V2
  `21.8385/13.3358/35.9990/45.1160/15.8179%`; IDSW was 1,997 and 9,915.
  These are internal validation-scope TrackEval results, not screening or
  official-test results.
- Root cause: correcting S-phase temporal/history evaluation does not
  restore expression correspondence; the fixed emission remains recall- and
  multi-positive-limited, with substantial inactive acceptance. Status is
  `L89E_COMPLETE_ZERO_TRAINING_PHASE_CONSISTENT_REPLAY /
  semantic_gate_fail / STOPPED_PENDING_SUPERVISOR_REVIEW`.
- Preserved outputs: Stage-S attempts 1–4, merge attempts 1–2, shortlist
  attempts 1–4, dev/internal full-video and TrackEval outputs, fixed semantic
  output, and all reports under `reports/l89e/` and `outputs/l89e/`. The fixed
  manifest SHA remains
  `06da458b09aa3e61ce30a4f8b58a85ac31ef1a5a10d269abd64ae41cffd127fa`.
  Flags remain `zero_training=true`, `screening_gt_used=false`,
  `official_test_labels_read=false`, `ordinary_mot_ovmot_touched=false`.
- The only next action is supervisor review of a separately authorized
  absence/volume-calibration study. Do not extend L89/L89C/L89E, alter the
  bank, or touch production MOT/OVMOT.

## R0A — safe target contract, dense native indexes, and implementation smoke — 2026-09-09

- Hypothesis: isolate only allowlisted V1/V2 target records, repair the
  frame-specific target contract, and construct native-frame R0 data without
  parsing forbidden official-eval payloads.
- Safe-source retry3 passed: lexical monolithic-JSON isolation deserialized
  only allowlisted video values and recorded
  `forbidden_payload_deserialized=false` and
  `official_test_labels_read=false`.  The exact 5,314 sparse fit rows matched
  the authoritative frame map with zero sentence/target/missing-query/invalid-
  frame mismatches.
- Dense native-frame index retry2 and its contract audit passed for V1/V2;
  all L69 rows and duplicate candidate indices were retained.  The geometry
  `image_hw` and visual-cache dataset/group-key contracts were corrected in
  R0-only files.  Retry1 dense/smoke failures remain preserved with their
  first trace; retry2 is authoritative.
- Real R0 contract smoke retry2 passed on one V1 fit frame: loss
  `3.555824041366577`, 108/108 finite nonzero trainable gradients, frozen
  GroundingDINO, no tracker optimizer parameters, strict reload max diff `0`,
  and temporary visual item removed.  This is implementation evidence only;
  no semantic, screening, official-test, HOTA, or production MOT/OVMOT result.
- Next action: commit/freeze the R0A safe code and construct the registered
  query-independent visual cache on `/data2`; no old asset or production
  entrypoint is modified.

## R0B — grouped DDP driver repair and formal legal evidence — 2026-09-09

- Hypothesis: the old Q1 query-at-a-time driver was the actionable R0 training
  contract bottleneck. A frame-centric grouped driver with complete candidate
  sets, Q=8, deterministic strata, and DDP accumulation should make the
  registered R0 experiment executable without changing the frozen bank,
  tracker, detector, or ordinary MOT/OVMOT.
- Preserved/disqualified Q1: PID 39571 was stopped gracefully; the old partial
  `outputs/r0/train/v1_formal_retry1/` remains historical and is not
  selection-eligible.
- Repairs: grouped Q8 schedule, video-local tiles, world4 DDP, accumulation 4,
  frozen detector/language inputs, complete candidate rows, present-uncovered
  masking, and resumable `locatemot-r0-tcgh-checkpoint-v2`. The legal GT
  materialization gap was repaired only in the R0 wrapper by recording and
  skipping missing annotation rows; frozen GT was not edited.
- Formal training: V1 completed 12 epochs/3,058 optimizer steps and V2
  completed 12 epochs/3,046 steps. Both were finite with nonzero gradients,
  strict reload, all four strata and both domains. Checkpoints 2/4/6/8/10/12
  are retained. Formal jobs used the free physical GPUs 4–7; physical GPUs
  0–3, which had unrelated processes, were not touched.
- Legal dev selection: V1 shortlist epochs 12/6 and V2 12/10; full dev
  TrackEval selected epoch12 for both. Dev HOTA was V1 22.8681% and V2
  19.5801% for the selected checkpoints.
- Internal validation-scope TrackEval: V1 0004/0018 HOTA 26.4582%, V2
  0016/0017/0020 HOTA 15.6097%. These are not screening or official-test
  results. No screening/official labels were read, and ordinary MOT/OVMOT,
  UIDM, L69 and production entrypoints were untouched.
- Outputs: training remains under `outputs/r0/`; large dev/inference/TrackEval
  products are under `/data2/usr_for_deadline/locatemot_r0b_*` because `/data1`
  was nearly full. No raw/dense debug cache or detector copy was created.
- Status: `STOPPED_PENDING_SUPERVISOR_REVIEW`. This is valid grouped R0
  implementation plus legal dev/internal TrackEval evidence, not final
  ordinary-RMOT completion. The only next action is supervisor review and a
  separately authorized follow-up.

## R1 — aligned track conditioning legal-dev contract — 2026-09-10

- Hypothesis/scope: test a zero-initialized RMOT-only residual conditioned on
  aligned frozen L89E Stage-S Z0/Z1/Z4 states and causal observation history;
  the anchor, L69 bank, ordinary MOT/OVMOT, UIDM and all production paths stay
  frozen. This is not final RMOT completion.
- Read-only legal-dev audit: frozen epoch-004 Rule-B anchor selection was
  rechecked. The epoch-004 presence-only miss is `3/400=.0075`, below the
  preregistered `.05` activation threshold, so the formal R1 config freezes
  `presence_residual=false`. Existing full-video artifacts have no candidate
  score arrays for a valid EXACT_INDEX_DEDUP replay; the conservative RAW
  policy is therefore frozen without claiming a dedup comparison.
- Preserved evidence: `outputs/r1/audit/legal_dev_selection_attempt1/` and
  `reports/r1/R1_LEGAL_DEV_SELECTION.md`; no training or new checkpoint was
  created by the audit. The next action is the bounded wiring/reload smoke;
  formal fit remains conditional on that implementation gate.

## R1 — formal fit and legal-development replay — 2026-09-12

- Hypothesis/scope: a zero-initialized aligned track-conditioning residual
  over frozen L89E stage states, causal observation history, language-directed
  R0 detail and complete L69 candidate sets could improve expression-to-track
  correspondence without changing the anchor, bank, tracker or ordinary
  MOT/OVMOT paths.
- Formal fit completed in isolated worktree
  `/data1/LWR/vranlee/SERVER_ONLY/avis/LocateMOT_R1`: six epochs, 2,250
  optimizer steps, all finite with nonzero gradients, strict reload for eight
  V1/V2 checkpoints, frozen anchor, and no persistent raw/dense debug cache.
- Prediction-only replays for legal-dev V1 videos `0008/0010/0020` and V2
  videos `0000/0008/0009` completed before legal GT was opened. The aggregate
  contains 2,385 groups and eight internal TrackEval results. Registered
  selection chose epoch 4 separately for V1 and V2. Selected HOTA was
  `24.2345%` (V1) and `23.6542%` (V2), versus the L89E anchor's `28.7628%`
  and `21.8385%`; this is not a stable breakthrough and remains legal-dev
  evidence only.
- Fixed 16-calibration/24-validation diagnostic retained all 2,435 L69 rows
  and 40 keys. R1 candidate-only recall/precision/FP-frame/pred-positive/
  hard/multi were `0.1000/0.1842/0.7750/0.5429/0.7727/0.1818`; the frozen
  Rule-B replay emitted zero rows for both anchor and R1 on this slice. The
  lower volume is a recall/multi-positive collapse, not a semantic repair;
  no threshold or NULL rescue was attempted.
- Status: `R1_NO_BREAKTHROUGH / STOPPED_PENDING_SUPERVISOR_REVIEW`.
  Outputs and failed technical attempts are preserved under `outputs/r1/`
  and `reports/r1/`. Flags remain
  `screening_gt_used=false`, `official_test_labels_read=false`, and
  `ordinary_mot_ovmot_touched=false`. The only next action is supervisor
  review and explicit authorization of one separate follow-up; no automatic
  continuation, screening, official test, or production merge is allowed.

## R1 — fixed semantic diagnostic closeout — 2026-09-12

- The fixed-slice evaluator used the native L62 order and attached labels only
  after row/feature construction and scoring. It used the frozen Rule-B values
  `candidate_threshold=1.0`, `presence_threshold=0.5`, `null_margin=0.0` and
  did not fit or tune a new threshold.
- This is a diagnostic, not a preregistered semantic gate and not HOTA. The
  authoritative files are in `outputs/r1/eval/fixed_semantic_attempt1/`; the
  corresponding report is `reports/r1/R1_FIXED_SEMANTIC_DIAGNOSTIC.md`.
# 2026-10-01 — new-server restoration started

- Hypothesis: restoring public V1/V2 data and exact base inference dependencies enables a new, attributable correspondence experiment; extending the failed R1 sidecar is unjustified.
- Evidence: fetched all remote branches; latest remains R1 `99090c8`. Read chronological historical endings and R1 reports: V1 legal-dev regression, limited V2 gain, fixed-slice score mismatch/output collapse. Old trained checkpoints are unavailable.
- Changes: started resumable official KITTI S3 **direct** download (15,813,146,295 bytes); downloaded and CRC-validated V1 expression/labels archives; isolated local data-tools environment. Full inventory/short plan: `reports/RESTORE_STATUS_20261001.md`.
- Failure/repair: shell sandbox lacks bubblewrap; execution requires tool escalation. Standard venv lacks ensurepip; local virtualenv bootstrap succeeded. Neither is a scientific result.
- Results: public V1 annotations recovered; KITTI/V2/runtime/project-generated assets pending. No historical checkpoint/bank reproduction claimed. Keep recovery scripts and new provenance; preserve migration AGENTS.md edit.

## LocateMOT-U U0 — unified specification and reference audit — 2026-10-01

- Hypothesis: a single specification-conditioned foundation can absorb the
  reusable persistent-query, temporal-language, open-vocabulary and reliability
  ideas from the ten requested repositories, provided the legal video boundary
  and one-checkpoint/shared-state contract are frozen before training.
- Code: cloned and shallow-audited MOTR, TransRMOT, TempRMOT, DKGTrack,
  FlexHook, iKUN, OVTR, COVTrack, ReferDINO and Open-GroundingDino; created
  `UnifiedSpecTrack` U0 shape skeleton, shared legal-scope guards, task/sample
  contracts, loss/runtime interfaces, parameter-sharing audit and reproducible
  audit/sanity tools. Exact source recipes and commit IDs are in
  `reports/unified/OPEN_SOURCE_TRAINING_AUDIT.md` and
  `outputs/unified/open_source_audit.json`.
- Data/protocol: froze V1/V2 fit/calibration/validation splits and reserved
  V1 `0005,0011,0013` plus V2 `0005,0011,0013,0019`; retained all 157 V2
  source target-ID-missing rows as unlabeled records. No official-test labels
  or screening ground truth were opened.
- Checkpoint/GPU: no LocateMOT-U checkpoint and no training run yet; GPU
  allocation was zero. External sanity was compile/help only (about 23 s wall
  time): 10/10 selected source files compiled, 1/10 entrypoints reached help,
  and the nine remaining failures are recorded dependency/version findings.
- Results: skeleton CPU forward finite with score shape `[1,100]`; legal guard
  smoke passed; skeleton non-head parameter fraction `0.9998259786` (not a
  final-model claim). No benchmark metric was produced. U0 scope/audit/sanity
  passed; U1 MM-GroundingDINO-B acquisition and proposal/expression gates are
  next.

## LocateMOT-U U1 — Swin-B foundation acquisition and adapter smoke — 2026-10-01

- Hypothesis: the audited GroundingDINO Swin-B CogCoOR weights can serve as the
  shared visual/specification foundation after selecting the checkpoint format
  compatible with the local MMDetection GroundingDINO implementation.
- Assets: acquired and hashed both the original release
  (`46270f7a822e6906b655b729c90613e48929d0f2bb8b9b76fd10a856f3ac6ab7`,
  938,057,991 bytes) and the OpenMMLab conversion
  (`55949c9c0f46339a73b415334765615d491ee6ed739ed3f568142b7fc5581143`,
  935,971,702 bytes). The MMDetection asset is the runtime candidate because
  its key layout matches the adapter; provenance is in
  `outputs/unified/protocol/foundation_asset.json`.
- Code/smoke: built the local Swin-B model (232,996,763 parameters), loaded the
  converted checkpoint with 8 expected initialization keys and 3 BERT metadata
  keys reported as non-fatal mismatch, then ran one CUDA label-free forward on
  legal V1 video `0004` frame `000063.png` using `car . pedestrian . cyclist .`.
- Results: 300 finite predictions, finite boxes and finite scores;
  `outputs/unified/u1_foundation_smoke.json` records the exact config,
  checkpoint hash, device and legal frame. This is a compatibility smoke only;
  no proposal recall, expression grounding metric or tracking result is
  claimed. Official-test labels and screening ground truth remain unread.
- Checkpoint/GPU/walltime: no LocateMOT-U checkpoint; one A800-equivalent GPU;
  model build/forward about 9 s. Next gate is label-free proposal and
  expression grounding on legal development frames.

## LocateMOT-U U1 — proposal gate diagnostic — 2026-10-01

- Hypothesis: the loaded Swin-B foundation would provide sufficiently localized
  open-vocabulary candidates for the shared semantic/tracking path on a fixed
  legal development slice.
- Protocol: 16 V1/V2 calibration/validation expression-frame rows, 50 target
  boxes, 300 candidates per row; model inference completed before each
  corresponding label file was opened. No threshold fitting, training,
  official-test labels or screening ground truth.
- Result after correcting the MMDetection BGR input contract: best-IoU
  `>=0.25` was `0.5400`, best-IoU `>=0.50` was `0.0400`, mean best IoU
  `0.2756`, and mean score at best IoU `0.01312`. Boxes and scores were finite.
  This remains below a usable proposal gate, so semantic-head/tracking training
  from this stream is disallowed. The initial RGB/BGR-swapped diagnostic was
  discarded and is not used as evidence.
- Diagnosis/next change: inspect prompt tokenization/category assignment, box
  decoding and proposal selection on this same legal slice. Preserve the split,
  rows, checkpoint and labels; do not tune a threshold or delete hard cases.
  Authoritative output: `outputs/unified/u1_proposal_gate.json` and
  `reports/unified/U1_PROPOSAL_GATE.md`.
- Selection-only check: retaining all 900 decoder queries raised IoU@0.50 to
  `0.1000` and mean best IoU to `0.3341`, so top-k selection contributes but
  does not repair localization. The top-900 output is retained separately and
  is not used as a training bank.
- A fixed class prompt (`car . pedestrian . cyclist .`) with all 900 queries
  reached IoU@0.25 `0.9000`, IoU@0.50 `0.1800`, mean best IoU `0.3635`; this is
  a useful direction for the repair but still below the stage gate and remains
  excluded from training. Output:
  `outputs/unified/u1_proposal_gate_generic_classes_top900.json`.

## LocateMOT-U U1.3 — proposal-selection repair diagnostics finalized — 2026-10-01

- Change: made the canonical 300-query output reproducible with explicit
  `max_per_img` and prompt metadata, and synchronized the report/status values
  with the corrected BGR run. Re-ran the 300-query baseline, the 900-query
  selection diagnostic, and the fixed class prompt (`car . pedestrian .
  cyclist .`) diagnostic on the identical legal 16-row slice.
- Results: the baseline stayed at IoU@0.25 `0.5400`, IoU@0.50 `0.0400`, and
  mean best IoU `0.2756`; retaining 900 queries reached `0.1000` at IoU@0.50;
  the fixed class prompt with 900 queries reached `0.1800` at IoU@0.50 and
  `0.9000` at IoU@0.25. All boxes/scores were finite, no rows were skipped,
  and no official-test labels or screening ground truth were read.
- Interpretation: prompt/category assignment and top-k selection are concrete
  repair directions, but the proposal gate remains below the stage criterion.
  No semantic-head/tracking training bank was created and no training was
  launched. The next experiment must keep this split/checkpoint fixed while
  inspecting prompt tokenization, category assignment and box decoding.
- Verification: unified modules/tools compile; CPU skeleton smoke is finite;
  parameter-sharing audit remains skeleton-only with non-head fraction
  `0.9998259786`; legal guard accepts V1 `0004` and rejects reserved `0013`.

## LocateMOT-U U0R/U1R measurement-contract repair — 2026-10-01

- Scope: resumed from `014b8a5ec81bb83c5993b56da171089f8a440a11` on
  `codex/u0r-u1r-foundation-repair-20261001`; no U2 work and no training was
  launched. The project root was verified as `/data2/user/LocateMOT`.
- Reproducibility hypothesis: a fresh checkout should import the unified data
  package using tracked source only. The first clone failed because the broad
  `data/` ignore rule also hid `locatemot/unified/data/`. Added a narrow ignore
  exception, committed the package source, pushed it, and re-cloned the remote
  branch at `26f9c989c8b13773d681bbe6962b77e73a0026bd`. Imports, skeleton smoke,
  and foundation smoke passed in the clone. Decision: keep the repair and mark
  U0 legal-scope reproducibility PASS.
- Checkpoint hypothesis: the MMDetection Swin-B conversion must have only
  documented non-core load mismatches. `tools/unified/audit_foundation_checkpoint.py`
  now requires the exact one training-only missing key
  `dn_query_generator.label_embedding.weight`, the exact three BERT metadata
  unexpected keys, and zero backbone/encoder/decoder/bbox-head/language-fusion
  missing keys. Audit result: `PASS_CONDITIONAL_TRAINING_ONLY_MISSING`.
- Preprocessing hypothesis: the formal runtime must be equivalent to the
  official MMDetection path. Ten deterministic legal images were compared with
  the same fixed generic vocabulary; all produced 300 finite predictions,
  maximum box difference `0.0`, and maximum score difference `0.0`. Decision:
  preprocessing contract PASS; the prior hand-built tensor smoke is not formal
  evidence.
- Split hypothesis: a deterministic video-disjoint split balancing expression
  volume, physical-target volume, and motion proxy is adequate before formal
  evaluation. `tools/unified/build_legal_split.py` froze v2 with seed `20261001`
  and split SHA
  `8ee806048ccbb227919f27edf68cf0bebc1fa50f824a8108878fc5f39fc6e1c0`.
  V1 fit/calibration/validation are `9/3/3` videos and V2 are `9/3/3`; V2
  source-target-ID-missing references remain `157` (fit only in this split),
  with no replacement or silent drop. Decision: freeze this split before any
  U1 adaptation or formal gate.
- Hardware/wall time: one CUDA device (A800-equivalent) was used for the
  checkpoint audit, smoke, and equivalence check; these short checks completed
  in under a minute each. Official-test labels and screening ground truth were
  not opened. External baseline execution remains incomplete and is the next
  blocking U0 task.

## LocateMOT-U U0R — TempRMOT legal-development execution — 2026-10-01

- Hypothesis: at least one published external RMOT method can be executed on a
  frozen legal-development subset without reading official-test labels or
  mixing its environment into LocateMOT-U.
- Protocol: copied TempRMOT at commit
  `6a65640d849fdee4a32bb055945ee34c3b0edeb1` into the isolated environment
  `/data2/user/reference_envs/temprmot`, loaded the released checkpoint
  `checkpoint_rk.pth` (SHA-256
  `55e9b2cecbc1590aface4c9d8639008e694e8c4556d1d3c27bf90549f408c7e8`), and
  ran the adapter on V1 validation videos `0012` and `0014`, one legal
  expression per video. TrackEval then evaluated the two generated sequences
  with HOTA, CLEAR, and Identity.
- Result: prediction and evaluation completed with 94 prediction rows and 93
  ground-truth rows. Combined HOTA was `94.679`, MOTA `98.925`, and IDF1
  `99.465`. Official-test labels, screening ground truth, and training were not
  used. The raw outputs and environment remain outside the repository;
  `outputs/unified/external_baselines/temprmot_legal_v1.json` records the
  reproducible metadata and `reports/unified/EXTERNAL_BASELINE_LEGAL_DEV.md`
  records the bounded scope.
- Compatibility work was confined to the isolated copy: a torchtext shim,
  current PyTorch `torch.load` behavior, disabling a redundant pretrained
  download, a CUDA scalar-type API update, and NumPy aliases in TrackEval.
  These changes do not alter the LocateMOT-U model or the frozen split.
- Decision: keep the TempRMOT run as real external-baseline evidence, while
  retaining `U0_EXTERNAL_BASELINE=INCOMPLETE` until a second independent
  method also produces predictions and evaluation. DKGTrack and FlexHook
  remain pending because their published weight links are not yet available
  through the isolated download path.

## LocateMOT-U U1R — formal generic and expression measurements — 2026-10-01

- Measurement contract: the frozen split SHA is
  `8ee806048ccbb227919f27edf68cf0bebc1fa50f824a8108878fc5f39fc6e1c0`; the
  foundation checkpoint SHA is
  `55949c9c0f46339a73b415334765615d491ee6ed739ed3f568142b7fc5581143`; every
  forward used `LoadImageFromFile`, `FixScaleResize(800,1333)`,
  `PackDetInputs`, and `DetDataPreprocessor` with BGR-to-RGB conversion.
- Generic protocol: one fixed query-independent vocabulary, all legal
  calibration/validation frames with materialized labels, physical units
  `(dataset, video, frame, track_id)`, Top100/150/300/900 diagnostics, and
  small/medium/large height strata. The run covered 4,201 unique images,
  33,526 physical GT units, and 5,992 unique physical objects. Top150 recall
  was V1 `0.0803` and V2 `0.0669` at IoU 0.50 (V1/V2 IoU .25: `.5309/.4940`,
  IoU .75: `.0045/.0062`). Decision: formal generic gate FAIL; this is below
  the `.80` adaptation early-pass, so a detection adaptation probe is required
  before any grounding adaptation.
- Expression protocol: every legal calibration/validation query contributed
  one deterministic middle labeled frame and, when available, one first empty
  target frame. Units were `(dataset, video, query_id, frame, target_id)`;
  source-missing references stayed in the source denominator. The run covered
  7,160 query-frame rows and 7,199 valid target units with zero source-missing
  units in this calibration/validation sample. Top1 IoU .50 recall was V1
  `0.0030` and V2 `0.0028`; AP50 and empty-frame FP/query are recorded in
  `reports/unified/U1_EXPRESSION_GROUNDING_GATE.md`. Decision: formal
  expression gate FAIL; no expression adaptation has started.
- No official-test labels or screening ground truth were read. No U2 or
  universal track core work was started. The external-baseline state remains
  incomplete because only TempRMOT has real predictions/evaluation; DKGTrack
  and FlexHook public Baidu links require an authenticated download path.

## LocateMOT-U U1-A — protocol correction, valid adaptation, and stop — 2026-10-01

- The first eight-process launch was invalid because MMEngine's default
  `--launcher none` left the processes outside DDP; it was stopped before any
  formal evidence. A later exploratory FP32 run also used `frozen_stages=1`.
  MMDetection's Swin implementation shows that value freezes patch embedding
  and stage0 only, while the registered contract requires stages 0 and 1.
  Those exploratory metrics are excluded from the final record.
- Corrected hypothesis/protocol: a fixed legal-fit generic stream can repair
  the foundation proposal localization while freezing Swin stages 0/1 and
  BERT. The legal-fit stream has 3,606 rows and fixed `label 0 -> car`; the
  split, vocabulary, official `FixScaleResize(800,1333)` pipeline, and all
  denominators stayed unchanged. The valid run used
  `frozen_stages=2`, BF16 AMP, eight-GPU PyTorch DDP, batch 2/GPU, and no
  accumulation (global batch 16). The logged peak was 5,945 MB/GPU. A two-step
  smoke passed before the full run.
- A batch-2/two-step-accumulation attempt passed a one-step smoke but failed in
  the full run with PyTorch's `expect_autograd_hooks_` DDP reducer assertion.
  The smallest working fix was accumulation 1; this kept the recommended
  microbatch 2 and did not alter the official input resolution. This failure
  and the corrected smoke are retained in the adaptation report.
- The valid schedule completed epochs 1, 2, 4, and 6, each formally evaluated
  on the unchanged 4,201-image, 33,526-physical-unit scope. Top150
  Recall@IoU0.50 (V1/V2) was foundation `0.0803/0.0669`, then
  `0.8729/0.8515` (epoch1), `0.8873/0.8766` (epoch2),
  `0.8945/0.8843` (epoch4), and `0.8938/0.8815` (epoch6). IoU0.25 reached
  about `.99` in both domains, while epoch-6 IoU0.75 was `0.7215/0.6119`.
  Size-stratum IoU0.50 was small `0.9136`, medium `0.9445`, and large
  `0.7085`.
- Training logs contain occasional `grad_norm: nan` values, but loss values
  remained finite, all six checkpoints were written, and every formal shard
  produced finite predictions. This telemetry is reported rather than
  filtered. The final epoch-6 checkpoint SHA-256 is
  `a1ff39f9c0bd59a5fd2b14968a528fc11e882b8fa6a3e85a9b64253a1cdc5ccc`.
- Decision: the valid V1/V2 generic gate did not reach `.90/.90` after the
  registered six-epoch maximum. Expression adaptation was not started because
  it is downstream of the generic gate. Added the corrected adaptation report,
  failure decomposition, and four valid formal JSONs. Final status is
  `LOCATEMOT_U_BLOCKED_GENERIC_GATE_AFTER_MAX_ADAPTATION` with
  `STOPPED_PENDING_SUPERVISOR_REVIEW`; U2 was not launched.

## LocateMOT-U U1R2 — measurement repair, class-aware refinement, and foundation completion — 2026-10-01

- Contract hypothesis: the remaining U1 failures were caused by a measurement/data contract mismatch rather than a need to replace the MM-GroundingDINO-B foundation. The source `labels_with_ids` fields are normalized top-left `xywh`; the repaired formal gates preserve source extents and use Top150 IoU.50 for generic proposals and Top100 IoU.50 target coverage for expressions. The previous U1-A-v1 files remain preserved as historical artifacts and were not overwritten.
- Coordinate/source audit: legal calibration/validation contains 33,526 physical GT boxes. 195 V1 boxes extend below the image at `y1>height` (166 large, 27 medium, 2 small); no x-side overflow exists. Same-frame geometry matching to official KITTI labels gives mean IoU `0.7414883`; Refer and official track IDs are separate namespaces. Decision: retain source extents and recover classes by geometry; do not clip evaluation GT.
- Data hypothesis: a class-aware physical-frame union can repair localization while removing V1/V2 duplicate training. The fit stream contains 2,545 images and 10,502 boxes; 3,424 V1/V2 overlaps were removed, 580 conflicting boxes and 21 missing class matches were recorded, and observed classes are car 6,787, pedestrian 3,700, cyclist 13, van 2. Decision: retain the stream with fixed class policy and explicit low-confidence audit.
- Numerical stability: the prior BF16 log has three finite-loss `grad_norm:nan` records (epoch/iter 1/50, 3/100, 6/100). A fresh FP32 one-step audit from epoch4 found 981 trainable parameters, total loss `40.27210998535156`, zero nonfinite gradients, and zero no-gradient parameters. Decision: R2 uses FP32/TF32, no AMP/BF16.
- Generic refinement: one bounded FP32 epoch from the frozen epoch4 checkpoint passed the corrected gate: V1/V2 Top150 IoU.50 `0.982716/0.971837`; small/medium/large `0.937467/0.989984/0.997825`. Decision: freeze the R2 epoch1 generic checkpoint and skip epoch2/mixed fallback.
- Expression hypothesis: whole-expression positive spans with all referred boxes, legal empty frames, and preserved V2 missing-target references can add language-conditioned candidate coverage without erasing generic proposals. One FP32 epoch completed from the selected generic checkpoint. Expression Top100 IoU.50 is V1/V2 `1.000000/0.961367`; the same checkpoint's generic regression is V1/V2 Top150 IoU.50 `0.986592/0.958739`.
- Language audit: on exactly 500 legal validation same-image pairs with a same-video sentence from a different target set, true versus shuffled Top10 recall is `0.959333/0.871133` and AP50 is `0.736901/0.491159`. Both strict comparisons pass, so language conditioning is measurable rather than generic-box-only behavior.
- External evaluator: synthetic perfect/empty/ID-switch and cross-evaluator arrays pass with maximum difference `0.0`. TempRMOT evaluated videos 0012/0014 occur in its declared training pool; its HOTA 94.679 is reclassified as `PIPELINE_SANITY_ONLY_TRAINING_OVERLAP`.
- Decision: the final shared expression checkpoint satisfies all U1 hard gates. `official_test_labels_read=false`, `screening_gt_used=false`, V2 source-target-ID-missing references remain explicit (157, silently dropped=false), and no U2/U3/U4 work started. Set `LOCATEMOT_U_READY_FOR_U2` and stop pending supervisor review.

## LocateMOT-U U2 — shared persistent track core and portability stop — 2026-10-02

- Provenance: working branch `codex/u2-shared-persistent-track-core-20261001`,
  authoritative local base `56507f4aa9a12e76d6f9f6702f0f0b38163666ec`, and
  immutable U1 foundation SHA
  `4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26`.
- Implementation: replaced the U0 tracking placeholders with one shared U2
  core (`PerceptionBridge`, independent identity/semantic candidate encoders,
  `UnifiedTrackState`, four-layer identity set interaction, GRU identity
  memory/anchor/EMA, association/new/no-match, lifecycle, motion, Hungarian
  rollout, and bounded OVMOT residual). The runtime is
  `locatemot/unified/runtime/u2_online_tracker.py`; no RMOT or prompt tracking
  was launched.
- Contract evidence: `tools/unified/u2_causal_contract.py` passed birth,
  occlusion, no-future, MOT specification invariance, OVMOT isolation, and
  online-state save/reload. Actual-model sharing audit passed with
  `shared_non_head_fraction=0.9378111093950003`.
- U2-P data manifest SHA-256 is
  `575bd349e00bab23634c6929c04513149afec838de419bb3f1a53469fafc7fda`.
  KITTI legal development portability passes (V1/V2 Top150 @.50
  `0.986592/0.958739`), while BDD100K, DanceTrack, MOT17, MOT20, TAO, and
  C-TAO are missing locally. Candidate-cache SHA is `not built`.
- Training command: not launched; GPU count `0`, optimizer steps `0`, wall
  clock training runtime `0`. Simple MOT, MOT-only, and joint OVMOT metrics
  are intentionally `null` because the registered portability protocol is
  incomplete.
- Decision: `LOCATEMOT_U_BLOCKED_U2_PERCEPTION_PORTABILITY`; stop pending local
  acquisition/audit of the missing streams. Official test labels and
  Refer-KITTI official evaluation labels remain unread.
- Validation note: the 2-process CUDA forward/backward smoke passed with the
  same MOT task on both ranks. An 8-process smoke could not initialize because
  all eight A800 devices were occupied by an unrelated process; it was
  terminated before any training step and is not counted as a pass.

## LocateMOT-U U2 — acquisition recovery — 2026-10-11

- Revalidated local descendant `666ab44` and the actual U1R2 checkpoint SHA
  `4efc35ce70be5cddf2c4d9a0f91bc4edef3e9ec5277a819211ba4ac42e0d7d26`.
  U1 references and the checkpoint remain unchanged.
- Observed MOT17/MOT20 extraction counts of 33,810/13,430 files, with nested
  release roots. TAO's four volume roots exist but contain zero files; no
  training or download worker was live. The old download module spawned
  workers while being imported by each child, recursively launching itself.
- Replaced that launcher with one guarded entry point and four thread workers
  running bounded download clients. Keep the registered four-volume assignment,
  prioritize modal train/validation inputs, validate downloaded sizes, and
  record actual per-file SHA-256 for source provenance. A fresh direct-network
  README transfer passed: 9,189/9,189 bytes, so existing Baidu authentication is
  usable. This does not imply that the dataset or portability gate is complete.
- Identified a separate eligibility bug: the old portability artifact could
  return PASS solely from directory availability while all new-domain metrics
  were null. Correct the audit and eligibility calculation before any training;
  metric definitions, split rules, thresholds, and seeds remain unchanged.

## U2 frozen-foundation MOT portability — 2026-10-11

Hypothesis: the selected U1R2 foundation may need bounded adaptation on crowded/non-road pedestrian domains. Registered ByteTrack inclusive train-half ranges, FRCNN physical-video deduplication, `person .`, positive mark-1/class-1 GT and unmodified source box geometry before measurement. The loaded checkpoint SHA matched the immutable U1R2 reference, with no missing/unexpected weights.

Ran all 7,115 dev frames on GPUs 1–7 in one parent launch/wait command, leaving an externally occupied GPU 0 alone. MOT17: 2,652 images / 53,890 GT; Top150 0.787233, Top300 0.806253. MOT20: 4,463 images / 615,137 GT; Top150 0.731010, Top300 0.884673. Exact coverage and monotonic count checks passed; partial measurements are rejected. MOT17-02/04 and crowded MOT20-03/05 dominate the shortfall. Retain the measured failure; do not alter visibility filtering, vocabulary, frame scope or gates to improve the reported values.

Track training is held. At most two epochs of mixed U2-P perception adaptation are justified after the missing domains/protocols are ready, with U1 replay and full retention gates. MOT20 Top300 has not met the preferred 0.90 or been accepted as a runtime override. TAO four-volume acquisition/extraction continues. Cloud BDD resources are static detection rather than MOT 2020. Started official DanceTrack train/val and C-TAO base-only acquisition, pinning release revisions; aria2 TLS failed, curl direct HTTP-206 probe succeeded and curl replaces it. No tracker or RMOT/prompt training was launched.

U2-P follow-up: C-TAO base-only source downloaded and inspected: 490,210 train images, 1,489,637 base-category boxes, 500 videos; supervision frequencies f/c only and no validation paths. LVIS v1 full vocabulary frozen (1,203 names; 866 f/c base, 337 rare novel). Native MMDetection chunking passed a real TAO-image smoke with 38 chunks and exactly one backbone call, followed by class-agnostic IoU-0.8 merge and global Top300. This is runtime evidence, not a substitute for full TAO coverage. Original modal TAO validation has 36,375 images / 113,112 boxes; 314 boxes lack a LVIS-v1 synset mapping, so the benchmark conversion/audit must explicitly account for them. Original BDD MOT 2020 image host dl.yf.io is live; started eight train/val archives (38,886,924,336 bytes) with original published MD5 checks, four bounded workers and eight connections/file. Official 2021 tracking-label S3 URL returns 404; keep BDD labels missing while locating a valid original release.

## U2 TAO/C-TAO portability registration — 2026-10-12

Continue from the published full MOT-half failure measurements, preserving the immutable U1R2 checkpoint. Plan: freeze all official TAO validation images plus every original annotated C-TAO training-audit frame; preserve the full dense C-TAO set for later identity training. Use all 1,203 LVIS-v1 names, the already verified 32-name chunks, one image backbone extraction, class-agnostic IoU-0.8 merge and global Top300. Count all original modal TAO GT including the 314 annotations without a v1 synset mapping; report those separately with source-frequency grouping rather than dropping them. Wait for CRC-completed source archives internally in the single evaluation launcher while downloads/extraction continue. In parallel, finish Dance extraction and locate original BDD tracking labels. No track training is eligible yet.

## U2 data completion and full portability launch — 2026-10-12

Hypothesis: recovering the original data and measuring complete dev scopes will distinguish foundation domain failure from incomplete measurement. DanceTrack download/CRC extraction completed; preserved 32/8/25 project videos and exact 33,772/8,024/25,508 frames. TAO modal train/val and all 490,210 C-TAO training images now pass availability coverage (six available streams, BDD pending). Frozen full TAO validation (36,375 images, 110,055 base / 3,057 novel GT) and original C-TAO train audit (17,131 images, 53,263 base GT); retain all 314 unmapped original TAO GT units. Immutable-foundation real smokes passed in both chunked domains and DanceTrack; full audits run on GPUs 1–6 and 7 respectively. No partial recall promoted; MOT gates still fail and track training remains held.

BDD root cause: old official annotation URLs are dead and the user cloud BDD folder is static detection. Found a pinned HF archive whose directory contains all 1,400/200 raw tracking JSON videos and validation images. Download/hash/selected-member CRC extraction is active; source labels have not yet established availability. Added original toolkit category/ignored/crowd and inclusive Box2D handling; raw-label full-scope validation will precede BDD measurement. Original seven train-image archives continue downloading, with an internal blocking extractor. Keep these recovery changes; no learned model, seeds or gates changed.

## U2 BDD raw tracking recovery validated — 2026-10-12

Recovered and SHA/CRC-validated the pinned mirror: all 1,400 train and 200 val raw JSON videos; every 39,973 validation image matches the original release ZIP member size/CRC. Frozen 278,079 train / 39,973 val frames and the fixed eight-category vocabulary before inference. Full official-reference comparison covers all validation objects: 442,814 positive boxes, 18,625 crowd and 4,073 official ignored-category objects. First strict comparison exposed a 2.84e-14 rounding difference from direct x2+1; changed arithmetic to the official xywh reconstruction order, then every positive box/category/ID matched exactly (no tolerance relaxation). Keep the correction. Full BDD portability is queued after the DanceTrack worker on GPU 7 with a real-image smoke first. The first original train-image archive is CRC-extracted (200 videos, 39,418 frames); remaining train-image downloads/extraction continue. Numerical MOT gate failures and track-training hold remain unchanged.

## U2-P bounded perception adaptation preparation — 2026-10-12

Plan before any adaptation result: use a fixed 20,000-anchor replay epoch, maximum two epochs, seed 20261012. U1 generic/V1-expression/V2-expression proportions 0.10/0.15/0.25; BDD/Dance/MOT17/MOT20/TAO base detection each 0.10. Stratified video sampling covers every training video where the allocated budget permits; cycle deterministically when replay pools are smaller than their allocations. Preserve original U1 replay boxes/phrases; use only official train or registered MOT-half train frames. TAO class supervision uses mapped LVIS-v1 f/c categories only, with novel/unmapped annotations excluded from class and identity supervision. Encode detection classes as positive phrase spans through the existing native ODVG grounding path. Empty positives retain a fixed absent base-class caption. Keep FP32, decoder/multimodal LR 1e-5, Swin stages 2/3 LR 1e-6, BERT and Swin stages 0/1 frozen. Do not start optimization until all source images and full F0 domain audits are complete. Any adapted checkpoint must pass the unchanged U1 retention and complete portability gates; no tracker/cache before that.

Implemented the fixed replay preparator, checkpoint-compatible frozen-BERT model, guarded eight-GPU trainer, explicit candidate-SHA evaluation, crowded-K registration, and full-audit checkpoint selection. Config parsing exposed an inherited-variable NameError; use MMEngine's `_base_.train_pipeline`. Disabled aspect-ratio buckets because odd bucket tails can add singleton steps; ordinary batching preserves exactly 1,250 steps/epoch at global batch 16. Actual immutable checkpoint CPU loading matches every state key, BERT stays eval/frozen, Swin stages 0/1 stay frozen and stages 2/3 train. Native ODVG packing passed on eight real training-stream images and one empty-positive image; first check assumed a box wrapper but the native pipeline returns tensors, so corrected that check and reran. Reaggregated all 7,115 original MOT frames: every result remains identical. Mixed-checkpoint measurement/worker aggregation is rejected; F0 MOT20 Top300 0.8846728452 correctly fails the mandatory 0.90 override threshold. Original generic/expression/500-pair full U1 reference scopes and shard provenance validate. Keep these changes. Replay preparation session 81929 waits internally for actual BDD extractor PID 686873; training preflight reports incomplete BDD images, incomplete full F0 audits and pending full replay. No perception optimizer, track training, or candidate cache launched.

Scientific source backups use `/data2/user/locatemot_git_backups/u2_latest.bundle`; the adjacent `u2_latest.commit` records the exact original local HEAD preserved in each updated bundle. Original U1R2 history and the BDD recovery commits remain in this history.

Scope correction before any optimization: an epoch-1 retention failure rejects that checkpoint for tracking, but the specification still permits the second bounded perception epoch. Allow that optimizer continuation after all epoch-1 audits complete; both retention and portability acceptance thresholds remain unchanged, and no rejected candidate becomes track-eligible. The two-epoch/2,500-step cap remains fixed.

## U2 DanceTrack full measurement and bounded continuation — 2026-10-12

DanceTrack's original blocking audit completed: all 25,508 dev images, 225,148 positive source GT; Top150 224,318/225,148 = 0.996313536, Top300 224,627/225,148 = 0.997685967. Full frame coverage, registered GT count and immutable foundation provenance pass. Keep the complete measurement; it shows that the current foundation's failure is not universal across ordinary MOT domains. BDD's queued real-image check passed and its complete 39,973-frame audit began on GPU 7; the second original training-image archive completed CRC extraction (200 videos, 39,990 frames). No partial BDD/TAO values are promoted.

Plan: connect the already registered two-epoch trainer to complete ordinary/TAO portability, original eight-shard U1 generic/expression and fixed 500-pair audits, conditional MOT20 K300 registration, and checkpoint selection. One blocking controller will wait for the existing F0 audit/data-preparation parents, train only after all prerequisites hold, pause at each saved epoch for full audits, and stop after selection or the absolute two-epoch cap. Preserve exact worker partitions and resume only unfinished units; no metric, seed, frame scope or acceptance threshold changes.

All original BDD train-image extraction and replay preparation completed successfully during this stage. Required data availability is now 7/7; 1,400 train videos and every 278,079 training annotation frame have an image. The fixed replay is complete (SHA `1be4a6e2d03fd3f59aeeca8b931c3f638a2ca1db015ea7d54afa2ca725970947`), all 1,400 BDD and 500 TAO train videos represented; actual prompts use at most 21 tokens. Native eight-rank Gloo verification used the real 20,000-row ODVG dataset and actual configured sampler/batching: 2,500 rows and 1,250 batch-2 steps per rank, globally no duplicate/omitted row, all 300,070 positive boxes retained, 1,385 empty-positive rows retained, zero native box-filter removals. No GPU model or optimizer was created by that check. The first launch exposed a missing `.venv/bin/torchrun` entry point; use the same virtual-environment Python's `torch.distributed.run` module, rerun the targeted eight-rank check, and apply the identical launcher fix to the continuous trainer. Keep this fix; no package, seed, precision, LR or budget changes. FP32 training keeps the already registered U1 TF32 flags. All original eight-shard U1 scopes validate, and the original U1 checkpoint is correctly rejected as a completed 1,250-step U2-P candidate. Only incomplete full F0 BDD/TAO audits now hold optimization. The controller validates each existing parent's full artifact as it exits, then owns training and all audits; it never terminates unrelated jobs or advances through a failed/incomplete measurement.

Scientific source commit `cf41d7312e7046c10ebc802aa624d1be81e1a021` is backed up in the latest local bundle. Launched the continuous command with `CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/run_u2_perception_adaptation.py --tao-audit-pid 670232 --bdd-audit-pid 781721 --replay-pid 816267 --extraction-pid 625610`. Controller PID 854970 / exec session 11636 is confirmed live and has already validated the completed replay parent; it still waits for the two original full F0 audit parents. GPU count 8; optimizer steps 0; candidate cache not built. Clarified the existing native resume semantics after reading installed MMEngine: optimizer, scheduler and sampler epoch resume, while separately launched epochs use the same fixed base seed 20261012; worker RNG streams are not restored. This documents the configured epoch-boundary behavior, without changing seeds or training implementation. End-to-end training wall time is pending the blocking controller's completion.

## U2 real feature bridge and eight-GPU temporal validation — 2026-10-12

Hypothesis: attaching final-decoder query provenance before native chunk concatenation/NMS and sampling pre-text neck maps can supply correctly aligned identity/semantic features without changing the frozen detector's candidate distribution. Implemented `runtime/u2_grounding_candidates.py`: original boxes remain original-coordinate xyxy; identity sampling scales them into resized/padded image coordinates, preserving preprocessor padding rather than normalizing against only original image extent. The extractor reads no annotations and stores no GT identities. Actual checkpoint-backed GPU-0 validation (`CUDA_VISIBLE_DEVICES=0 OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/validate_u2_candidate_features.py --device cuda:0`, terminal session 84086, exit 0) passed native box/score/class bit-exact parity for MOT17-02 frame 000302 Top150 and one TAO validation frame under the complete 1,203-category/38-chunk Top300 protocol. Selected decoder features remain bit-exact after cross-chunk NMS; backbone executes once; fixed-box identity features remain bit-exact after changing vocabulary. Scope is an alignment contract, not a replacement for full-dataset recall. Keep the bridge; full cache awaits passing U2-P selection.

The eight-GPU temporal contract (`CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 OMP_NUM_THREADS=2 .venv/bin/python -m torch.distributed.run --standalone --nproc-per-node=8 tools/unified/u2_ddp_temporal_contract.py`, terminal session 18873, exit 0) passed both MOT and OVMOT. Two four-frame clips/rank (global 16) retain temporal gradients; identity GRU, set interaction and association gradients are nonzero, with semantic GRU/text gradients also nonzero on OVMOT. Parameters remain bit-exact across all eight distinct GPU ranks after scratch updates. This synthetic probe does not modify the foundation or constitute a formal association run; no tracking metrics are claimed. Keep the full-clip-in-one-DDP-forward contract for the eventual trainer. Evidence: `eval/ddp_eight_rank_core.json` and `eval/real_candidate_feature_contract.json`.

Confirmed GitHub publication of original source `0b6f1d4e20faed17bc105b67aef4505bcdc87ea2` at remote `5d79bfcc0b0b3f58594817dc727d5707bb1d6331` (verified branch and commit tree through the connected app). Corrected stale final-status fields that still called BDD missing and Dance unmeasured: all seven sources are available, Dance complete, BDD/TAO full audits remain pending. The existing blocking continuous controller remains live; no gates, seeds, candidates or active audit jobs were changed.

## U2 candidate-cache and ordinary-baseline execution plan — 2026-10-12

Read the real candidate bridge, frozen ordinary/TAO protocols, KITTI legal scope and TrackEval readers. The TempRMOT fork changes MOT file loading into expression-specific prediction paths; do not use that reader for ordinary MOT evaluation. Original KITTI tracking labels already exist under `/data2/user/reference_envs/kitti_tracking_labels_20261001/training/label_02`; no extra dataset download is required. Plan: pin an unmodified upstream TrackEval, freeze KITTI V2 legal fit/calibration/validation frame lists with original tracking IDs, implement a label-free video-sharded cache that requires the existing complete perception selection, and implement the IoU/motion/visual-cosine simple baseline with fixed parameters before any benchmark results. Keep supervision in separate source adapters, preserve all existing gates and running full F0 jobs, and test only the contracts needed for valid tracking/evaluation.

Pinned direct upstream TrackEval at `12c8791b303e0a0b50f753af204249e622d0281a`, verified unmodified, and recorded its six relevant readers/metrics. Frozen original KITTI V2 legal fit/calibration/validation covers 15 videos, 1,814/2,087/1,385 frames; never opened the four reserved video labels. Original KITTI and Refer tracking namespaces differ, so the earlier U1 generic result alone was incomplete evidence for ordinary tracking GT. Added the missing full original-GT portability audit over 3,472 dev images / 15,875 eligible car/pedestrian units with unchanged pinned TrackEval occlusion/truncation handling. Existing U1 references, all old domain scopes, .85/.90/.80 ordinary thresholds, .70/.60 OVMOT thresholds and the two-epoch cap remain intact. Actual F0 KITTI image smoke passed (session 37939 exit 0, native state keys all matched). Launched the full original-KITTI F0 audit on GPU 0: `OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/u2_mot_portability.py --protocol outputs/unified/u2/protocol/kitti_ordinary_portability.json --out outputs/unified/u2/perception/kitti_original_gt_f0_full --gpus 0`; parent 864092, worker 864093, blocking session 95238. The old full BDD/TAO jobs remain untouched. U2-P candidate evaluation now includes 76,068 ordinary images, plus all original 53,506 TAO/C-TAO and U1 scopes.

Implemented full label-free cache inventory and selected-checkpoint-gated 64-frame video shards. C-TAO dense and modal TAO train union contains 491,353 unique train images (including 1,143 modal images outside the C-TAO set), plus 36,375 TAO dev images. OVMOT storage assignment balances four volumes at 131,934/131,931/131,931/131,932 frames. Ordinary train/dev cache covers 396,865 images. Actual inventory verification (session 32148 exit 0) proves all 3,182 groups and all 924,593 physical images occur exactly once, with no train/dev image overlap and exact eight-worker group coverage. Cache generation remains conditional on recomputing the original complete selection; features are BF16, original boxes/scores FP32, semantic/decoder storage is shared, and supervision stays separate with no automatic novel labels.

Implemented the frozen simple baseline and official full-sequence metric/export path. Registered parameters before benchmark results: IoU/motion-IoU/id-cosine .25/.35/.40, margin .45, distance 2, birth .10, appearance EMA .90, max_age 30. All input candidates participate in matching, low scores can recover IDs, and no motion-only box is emitted. Synthetic causal/export/metric validation (session 40233 exit 0) passed all five upstream readers: perfect HOTA/DetA/AssA/IDF1 1 and IDSW 0, while swapped MOT17 identities give DetA 1, AssA .4, IDF1 2/3, IDSW 2. BDD inclusive extents/aliases/ignored attributes and KITTI DontCare/MOT distractors are honored. No benchmark metric is claimed. The optional unrelated BURST import warns about missing tabulate; all required ordinary readers and metrics execute successfully, so no unrelated package repair is needed. Keep the cache and simple pipeline.

Launched one blocking follow-on queue: `OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/queue_u2_ordinary_baseline.py --wait-perception-pid 854970 --kitti-audit-pid 864092 --extraction-pid 625610`, controller 864319 / session 35256. It waits for genuine completion of the original-KITTI audit and the existing bounded perception owner, then builds the selected ordinary cache and measures all five simple-baseline dev sets. If the existing owner genuinely terminates at preflight before any optimizer because the newly supplied KITTI audit was still incomplete, the queue resumes only after that audit completes and the full unchanged preflight is READY. No live job is restarted on an observation timeout. Formal cache and formal track training are still unlaunched; the research goal remains active.

## U2-A temporal supervision implementation plan — 2026-10-12

The complete original-KITTI F0 audit terminated successfully (session 95238 exit 0): 3,472 images, 15,875 source GT units, Top150/Top300 recall .928503937/.940535433. The original perception owner and ordinary follow-on queue are confirmed live. Read the U2 temporal curriculum/teacher forcing/loss specification and actual core: query drop/injection is currently diagnostic-only, and motion/lifecycle assume consecutive processed frames despite required stride 1–4. Implement real training-state augmentation from past unmatched candidates, current-frame Hungarian GT targets and scheduled model-in-loop state assignment, the six registered identity/motion/lifecycle losses with explicit false-birth suppression, and physical frame-gap-aware state propagation. Use the same shared core, keep GT bookkeeping outside its runtime state, retain original gates and do not launch formal association optimization before selected caches and simple metrics exist. Validate crossing/occlusion, negative proposals, mixed teacher/model rollouts, connected temporal gradients, causality and task isolation before the 6,000-step run.

Implemented actual query-state drop and previous-frame hard-negative injection; external first-identity bookkeeping does not hide model switches or duplicate births. Registered six losses include positive column CE plus NEW reject BCE so false proposals (including the first frame) are not blindly taught births. Current-frame positive/distractor Hungarian and separate crowd IoA/min-height rules preserve official ignore semantics. Physical source-frame gaps drive motion, lifecycle and memory decay; expired IDs are retired before current matching. Batched association data transfers, GRU/birth updates and switch margins avoid per-track GPU/CPU synchronization without changing core parameters. Crossing/occlusion, false-birth/ignore gradients, mixed teacher/model rollout and connected temporal gradients pass; all existing causal/isolation/save-load checks pass.

Actual eight-GPU BF16 registered-loss proof initially failed at the Hungarian NumPy conversion (`TypeError: unsupported ScalarType BFloat16`, session 51784 exit 1; reproduced single GPU session 68632). Smallest fix: cast detached logits to FP32 only at the CPU assignment boundary. Single-GPU regression session 30093 and eight-GPU session 2469 passed: global16 clips, finite/nonzero candidate projection/identity GRU/pair/set/motion/alive/birth gradients, identical post-update rank parameters. Subsequent batched-transition/expiry edits trigger fresh targeted and dual-task/eight-GPU checks before publication. The first original-GT oracle check (53170 exit1) compared MOT reader XYWH to core XYXY; verified native source/reader format and fixed the oracle only. Session 16135 passed every positive box on five complete actual train videos: BDD203/1126 units, Dance703/4912, MOT17 301/8701, MOT20 215/9060, KITTI154/234. Scope is these five videos, not all train labels or benchmark results.

Added frozen-cache causal clip sampling, 6,000-step eight-GPU MOT-only entry, registered curriculum/teacher probabilities/seed20261012/dataset probabilities, full all-five dev evaluations at1000/2500/4000/6000, 500-step resume snapshots with optimizer and per-rank CPU/CUDA RNG, unchanged identity-benefit/HOTA gates and four-level checkpoint selection. Synthetic sampler/resume/selection proof passes: next stochastic update bit-exact after restore, train-only rank/step sampling, >2-point individual HOTA regressions rejected. Preflight (55952 exit2) correctly holds without the complete selected cache and all-five simple metrics; zero formal track updates. Launched one blocking follow-on queue: `OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/queue_u2_mot_only.py --wait-ordinary-pid 864319`; actual PID870157/session38396 waits for original ordinary owner, then launches exactly 6,000 MOT-only steps when prerequisites pass, without OVMOT. Original full F0/perception owners remain live and untouched. Keep implementation; U2/U3 and all trained metrics remain unproven.

The fresh batched-transition eight-GPU command (76774 exit1) passed both MOT/OVMOT temporal tasks first, then exposed a BF16 vector-index assignment mismatch: GRU output BF16 into FP32 persistent memory. Added explicit differentiable casts to destination identity/semantic memory dtype; no loss or parameter change. Run the actual single-GPU BF16 backward/active-state inference regression, then the full eight-GPU registered-loss proof again before publishing. The failure is explicitly recorded; old PASS artifacts do not certify the new code until the fresh proof succeeds.

Final current-source targeted regressions completed: session21978 exit0 (six-loss crossing/expiry, five actual-video GT comparison, sampler/RNG/selection), session1677 exit0 (post-dtype-fix temporal/causal/save-load and resume/selection), session87550 exit0 (fresh actual eight-GPU BF16 registered losses, all necessary gradients nonzero, both updates identical across ranks). The preceding dual-task eight-GPU FP32 phase of76774 passed for MOT and OVMOT; only its BF16 phase failed and was repaired. Single-GPU session57241 exit0 proves batched BF16 backward and nonempty-state causal inference. Keep these final implementations and proofs. No formal feature cache, learned MOT baseline or joint benchmark is claimed; the research goal remains active.

During verified publication/follow-on waits, original perception owner11636 reported BDD audit completion. The original BDD blocking session36974 terminated exit0; strict full measurement validation confirms all39,973 images,442,814 GT, coverage/protocol/F0 SHA. Hits363,947 Top150 (.8218958750175017),377,519 Top300 (.8525453124788286). BDD fails .85 minimum but exceeds .80 hard stop; no BDD K300 exception is allowed. All five ordinary F0 audits are now measured; bounded U2-P still waits for original full TAO/C-TAO audit. Updated canonical portability/status/reports and retain the same registered 2-epoch repair and six-thousand-step downstream gates. No formal track optimizer launched; keep result and publish this new full measurement.

## U2-B supervision/semantic/evaluation implementation plan — 2026-10-12

Previous goal turn made concrete progress: complete original KITTI/BDD F0 measurements, actual temporal supervision plus gated 6k MOT-only implementation, current eight-GPU BF16/causal/GT/resume proofs, source commits/backups and verified GitHub publications. Current source30c7a56 is clean; original bounded perception owner11636 and MOT follow-on38396 are confirmed live. Read OVMOT adapter, shared language bridge and U2 sections66–82. The OVMOT report incorrectly says TAO/C-TAO absent; all assets are present. Raw C-TAO interpolation resets frame_index (e.g. original frame0176=175, interpolated frame0177=1), and C-TAO/TAO numeric IDs occupy different source namespaces; physical filename chronology and original annotation UUID alignment are required. Plan: audit all dense/base supervision with no automatic rare/unmapped class or identity labels; implement shared-U1 BERT/projected full-vocabulary class features, semantic classification/consistency/gate losses and same-core OVMOT temporal rollout; pin official upstream TETA and implement complete original validation export/overall/base/novel reporting with explicit category mapping counts. Keep all ordinary loss weights and gates. Formal joint optimization requires selected MOT-only eligibility and is not authorized by contract fixtures or incomplete measurements; do not launch it now.

Full compiler failures were diagnosed sequentially without dropping units: camera timestamps require original index anchors (66320 exit1), LaSOT uses numeric filenames (second command terminal exit1), dense ArgoVerse frame_id compresses missing-image gaps (22205 exit1; 17/37 camera videos have nonconstant anchor offsets), and official dense annotations include nine zero-area boxes (97023 exit1). Retain source geometry/counts, use numbered filenames or timestamp interpolation between original TAO anchors, and leave zero-area GT at zero IoU. Final compiler29355 exit0 covers all500 train videos/491,353 images/1,489,990 unique base units; 52,911 base units deduplicate across sources, 353 modal-only base units remain. Novel/unmapped modal1,156/219 units never provide class/identity labels. The initial assumption of differing overlapping numeric IDs was not observed: all shared UUIDs have equal numeric IDs. Keep UUID alignment with original IDs retained separately. No validation/test labels enter training.

Actual F0 shared text extraction63671 exit0 yields1203x256 features,866 base/337 novel,38 native BERT chunks/max104 tokens/backbone0; F0 remains reference-only, not selected for formal training. Implemented existing-parameter semantic classification, final pair supervision, persistent-memory classification, consistency and gate regularization, with unknown candidate/row/contrast labels masked and certified negatives retained. Single80792 exit0 and actual eight-GPU7759 exit0 prove BF16 connected identity/semantic gradients, unknown birth-label exclusion, ordinary text isolation, full novel inference vocabulary and identical post-update ranks. This is one scratch update on synthetic track candidates with actual text features, not joint training. Ordinary temporal regression20603 and32602 exit0 passes unchanged six-loss values. Actual AVA/ArgoVerse/LaSOT full-video loaders67140 exit0 cover976/451/1169 frames and4074/1372/1169 units, all base-only/strictly increasing time. Deterministic joint task draws give6518 MOT/3482 OVMOT in10,000 steps; registered probabilities remain.65/.35.

Pinned unchanged official TETA b498aa87 and prepared complete LVIS-v1 TAO validation10545 exit0: all36,375 images/988 videos;112,798/113,112 original GT units and5473/5485 tracks map to v1;314 units/12 tracks explicitly unmapped by the official OVTrack synset conversion. Original v0.5 full-localization scope remains unchanged. TETA fixture22487 failed because OUTPUT_SUMMARY=False leaves upstream returned threshold uninitialized; use the official default summary=True with no source/metric modification. Fixture67046 exposed our bit-packed export IDs allocating huge dense upstream ID maps; stopped that owned synthetic command (exit143), removed the packing, and let official TAO resolve video-local IDs. Final61184 exit0 passes perfect100, wrong-class localization/association separation, identity-switch sensitivity and non-exhaustive unlabeled-object neutrality, with empty frames retained. No benchmark TETA/ordinary tracking score claimed, no formal joint optimizer or U3 launch. Keep all changes; original perception and downstream queues continue under unchanged gates.

Fresh ordinary eight-GPU BF16 regression71729 exit0 after the OVMOT changes again confirms all six registered loss gradients and identical rank parameters. Existing gate recipes, candidate caps, seeds, metrics and training scopes stay unchanged.

Current-source causal/state save-reload regression58539 exit0 passes on nonempty states, including ordinary text invariance and OVMOT raw identity/GRU isolation. No full candidate cache, simple tracking metric, learned MOT checkpoint or joint benchmark yet; continue the active U2 goal after publishing this completed source/supervision/semantic/TETA stage.

## U2-B gated joint execution plan — 2026-10-12

Previous goal turn made progress: full base-only dense supervision, actual shared F0 class features, same-core semantic/partial-label losses, official TETA validation conversion/export and actual eight-GPU proofs were committed3d2a9ec and published/verified as aff3b00. Current project identity/worktree are correct/clean; perception854970, ordinary864319 and MOT-only870157 are confirmed live. Read current baseline trainer/evaluator, both cache phases and U2 sections64–85. Implement exact4,000-step joint entry initialized with every selected MOT-only core tensor; global task.65/.35, all8 ranks/two8-frame clips, frozen foundation/BERT and separate5e-5/1e-4 core/adapter groups. At each1,000-step boundary evaluate every ordinary dev sequence plus all TAO validation through official TETA; reject any individual HOTA or AssA loss>.01 and enforce TETA34/AssocA37. Add exact RNG/optimizer resume and one blocking downstream owner that waits for real MOT-only completion/eligibility before OV feature-cache/class-text construction and joint launch. Actual semantic fixtures do not authorize a formal run. The full rank-zero metrics can exceed default NCCL collective timeout; extend only the collective wait timeout, with no change to data/metrics/optimizer/evaluation boundaries. Validate gate rejection, checkpoint initialization/resume and alternating-task eight-GPU execution before queuing.

Implemented the complete4,000-step joint trainer/evaluator and gated downstream owner. Registration/preflight55888 and29449 correctly return exit2 HELD with zero formal updates while selected caches/MOT-only are absent. Joint boundary85960 exit1 exposed .59-.60 floating comparison falsely rejecting the inclusive1pt threshold; MOT boundary4594 exit1 also exposed .4+.03 rejecting exact3pt gain. Compare the recorded decimal scores at the same literal.01/.02/.03 thresholds, with no epsilon/extra rounding tolerance or gate weakening. Final joint62479 and ordinary22178 exit0 pass inclusive boundaries, individual HOTA/AssA rejection, both OVMOT gates, refinement conditions and four-checkpoint selection. Original ordinary14478 regression also passed before this targeted numeric fix.

Actual eight-GPU BF16 alternating-task/init/optimizer proof14038 exit0 passed; strengthened1629 exit0 additionally restores the real scheduled teacher Bernoulli RNG for a bit-exact next update after actual joint checkpoint save/load. Both use all256-D core tensors,16 global clips/eight frames, actual reference-only F0 class text and synthetic tracking candidates; no formal optimizer exists. First whole eight-rank joint-evaluation fixture84831 exit1 failed at BDD predicted category person: the fixture incorrectly used the raw GT alias as detector vocabulary. Correct only the fixture to the existing registered pedestrian vocabulary; keep the actual BDD normalization, official reader, metrics and original scope unchanged. Run the whole entry again before queuing/publication.

Final whole joint-evaluation proof99228 exit0 executes the actual eight-rank entry, all five official ordinary readers, both base/novel TAO fixture videos and official TETA with8 CPU workers. Simulated interruption after completed video units resumes with zero core forward calls and identical full metrics. These scores are explicitly synthetic and are not benchmark evidence; no real tracking/foundation optimization or U3 readiness is claimed. Keep exact whole-entry implementation and all gates. The current source stage now has a single blocking joint downstream owner waiting for genuine completion of MOT-only870157; only a recomputed eligible selected reference permits full balanced OVMOT feature caching, selected actual class text, and the4k joint run. No live source jobs were restarted.

Launched that single blocking owner: `OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/queue_u2_joint.py --wait-mot-only-pid 870157`, actualPID883611/session32672. State confirmedWAITING_ELIGIBLE_MOT_ONLY; fullOVcache_started=False and formal_joint_training_launched=False. Keep all new implementations/proofs and publish the completed joint execution stage; the full U2 goal remains active with actual perception/cache/simple/learned/joint measurements unproven.

## U2 self-contained native checkpoint implementation plan — 2026-10-12

Previous goal turn made progress: scientific355c152 and exact eight-GPU joint execution/resume/evaluation proofs were committed, backed up and verified at GitHub c16cd98. Project root is correct and original perception/cache/baseline/joint owners are confirmed live. Read actual native runtime, U2 core and specification sections83–86. Final format must embed actual MM-GroundingDINO/shared BERT, tokenizer/config, every persistent core/head and task embeddings; the shape-only fallback is insufficient. Current core has no task embedding tensor. Add zero-initialized task embeddings only to the bounded semantic gate input; ordinary MOT gate remains exactly zero and raw identity/GRU remains task isolated. No formal track optimization has begun, so register this missing required component before selected caches/checkpoints. Implement native model reconstruction from embedded BERT architecture/tokenizer and strict embedded weights, with no external checkpoint/HF pretrained load. Validate actual F0 candidate/class-text bit-exact reload, causal state resume and same-core parameter sharing with a clearly reference-only random core pack; never treat it as a selected final checkpoint or U3 readiness. Actual final packing must recompute joint and perception eligibility and retain all original metadata/scopes.

Native reference-pack32496 exit1 exposed bbox-head constructor-derived fields saved after native configuration mutation;64857 exit1 then exposed the analogous denoising-query fields. Remove only those six constructor-derived entries before serialization; native construction recreates them.41966 exit1 passed strict complete foundation/core tensor reload but failed shared-BERT feature equality. Targeted31560 exit0 proves native from_pretrained uses SDPA while bare embedded BERT construction uses eager: max feature discrepancy1.144409e-5; current native output remains bit-exact with the immutable F0 text cache. Store and restore the actual native attention implementation rather than weaken comparison tolerance. Task-embedding joint eight-GPU60299 exit0 retains bit-exact next stochastic optimizer update, nonzero embedding gradient and ordinary adapter gradNone.35642 exit0 passes causal/temporal and CPU semantic regressions; strengthened38946 exit0 explicitly perturbs task embeddings and proves ordinary MOT unchanged/raw OVMOT identity isolation plus unchanged eligibility/initialization checks. No formal optimizer, reference, live owner, seed or gate was changed.

Attention attempt51064 exit1 still used eager because HF BertConfig.from_dict mutates the supplied dictionary and overwrites attn_implementation from kwargs;15925 verifies the actual packed class remained BertSelfAttention vs native BertSdpaSelfAttention. Pass a copied config plus explicit attention kwarg. Final native19361 exit0 proves all actual F0 foundation/core tensors,1203 class features, native MOT150/OVMOT300 candidate features and nonempty online state reload bit-exact, with external tensor/config/pretrained-HF calls prohibited. V1/V2 small deterministic generic/expression/true/shuffled integration outputs and metrics are bit-exact; this is a reference-only random core pack, not a selected final artifact or tracking benchmark. Task embeddings now use supplied zero weights without consuming RNG, preserving all subsequent shared-core initialization draws at the original seed. Final joint52310, causal/temporal/gate8930 and whole eight-rank official evaluation33800 exit0 pass. Targeted32182 recomputes actual parameter sharing after excluding shared detection/association/motion/lifecycle heads:236,751,992 shared non-head vs329,475 conservatively task-specific, fraction.9986102878>=.90. Only count metadata changed in the reference file; actual restored foundation/core tensors remain bit-exact. Formal packer recomputes original perception/MOT/joint eligibility, embeds original source/protocol/sampler/step metadata and remains pending actual selected-runtime retention. Keep all changes. No selected cache, real tracking benchmark, formal optimizer or U3 readiness yet.

Current CPU semantic18032 exit0 confirms task-embedding gradients and unknown-row/candidate masks after preserving initialization RNG. Store CPU evidence separately as ovmot_training_contract_cpu.json, retaining the historical GPU single-frame report. Updated eight-GPU joint evidence covers BF16 gradients and exact restore with current embeddings.

## U2 frozen text reuse for upcoming candidate caches — 2026-10-12

Previous goal turn made progress: actual native one-file F0/core reconstruction, task-embedding isolation and real parameter-sharing/retention proofs were committed2b775b7, backed up and verified at GitHub9c12586. Project root/clean worktree verified; original F0/perception and downstream owners are live and ordinary cache is still unlaunched. Read native GroundingDINO.predict: every full LVIS frame re-encodes the same38 BERT prompts although the language backbone is frozen; native prompt/token-positive-map preprocessing is also repeated. Implement worker-local fixed-text memoization by wrapping native language forward and prompt construction, preserving all native image/backbone/encoder/decoder/NMS operations and cloning returned tensors/maps before mutable native consumers. Apply only to upcoming full selected-foundation cache workers, without changing any running full audit or perception-training process. Measure actual FP32 F0 two-frame MOT and full1203-class OVMOT candidate/selected-feature equality, mutation isolation and actual BERT-call counts/timing; no approximate precision, cap, chunk, vocabulary, seed or scope change. Keep only if bit-exact outputs hold.

Initial frozen-text38589 exit0 proves actual FP32 F0 native/cached MOT150 and complete1203-class OVMOT300 two-frame candidate/feature parity, tensor/map mutation isolation and exact method restoration. Warm OVMOT elapsed5.208873→4.619842s (1.1275x observed throughput), with two-frame BERT76→38 calls and unchanged2 backbones/76 decoder calls. Implement only in upcoming selected-foundation cache workers; record source/proof provenance in spec and actual counters in all worker/full reports. Integrated80238 exit0 reconfirms full parity and executes the actual worker/serializer on explicitly separate two-frame reference inventories for both phases: serialized FP32/BF16 fields bit-exact, semantic/decoder storage shared, zero GT labels, completed-unit resume with no inference and identical unit manifests. Final warm OVMOT5.255846→4.638013s (1.1332x), MOT.188991→.180778s (1.0454x); shared-GPU two-frame observations, not full-cache speed estimates. Keep optimization; all actual source frames, chunk/candidate limits, precision, eight-rank assignment, four-volume distribution, original gates and running full jobs are unchanged. No formal selected cache, track optimizer, benchmark tracking score or U3 readiness exists yet.

## U2 selected-artifact finalization execution plan — 2026-10-12

Previous goal turn made progress: bit-exact real F0 fixed-text reuse and actual cache-worker serialization/resume were committed1ad7563, backed up and verified at GitHub4306013. Current project/worktree are correct/clean. Original joint blocking handle32672 is confirmed live and waits eligible MOT-only; no formal OV cache/joint has begun. Read U2 sections83–89/98, actual packer, small retention audit and joint owner. Implement a separate single blocking finalization owner waiting original883611 completion, then recompute full perception/MOT/joint eligibility, package the actual selected U2-B foundation/core, audit actual packed retention and unmodified selected-core causal/isolation/save-load behavior, and generate final readiness solely from current full artifacts/provenance. Reference-only F0 packs and scratch proofs cannot fulfill selected-artifact gates. Reject or retain exact held/blocked status when earlier gates fail; conditional U2-C remains a separate bounded task if eligible, never substitute a U2-B pack. Preserve all running owners and budgets. Use deterministic real candidate probes for actual learned heads; never modify selected weights to make runtime contract checks pass.

Existing historical two-GPU smoke had only one backward, while the requested two-step contract was missing. Updated the existing joint proof entry to accept expected world size2 or8 without changing its default eight-rank behavior; actual CUDA0/7 two-rank26053 exit0 now covers MOT/OVMOT updates and exact stochastic next-update checkpoint/RNG/optimizer restore. Fresh single-GPU8844 exit0 includes nonzero current task-embedding gradient and unchanged BF16 identity/semantic loss values; existing current-eight-GPU evidence stays intact. Reference selected-core-audit entry99248 exit1 exposed native candidate metadata strings being cloned as tensors; clone only tensor fields and retain metadata unchanged, then rerun targeted entry. This is an audit-interface repair, not a selected-weight, seed, gate or training-protocol change.

Selected-core entry48760 exit0 now passes four actual native F0 MOT frames and unchanged reference-core weights:150 births/repeat matches, one-frame occlusion/recovery, future-state independence, arbitrary MOT text invariance, raw OVMOT/fixed-pair GRU isolation and bit-exact state resume for both tasks on one core object. Readiness69971 exit0 tests the independent complete requirement set/material failure cases and the actual CLI returns2 HELD with missing full cache/simple measurements. Actual selected-audit CLI75746 exit0 proof rejects a reference-only random pack before cache read. Review found old small generic retention used only car rather than U1 original10-entity vocabulary; use original VOCABULARY in final audit and require that exact vocabulary in readiness. Full-vocabulary native/reference-pack96632 exit0 is bit-exact on both V1/V2; all original full U1 references are unchanged, and historical reference metadata now names the restricted car-only scope. Missing actual selected audits never become readiness evidence. Keep all repairs/entries.

Launched one blocking finalization owner with OMP_NUM_THREADS=2 TOKENIZERS_PARALLELISM=false .venv/bin/python -u tools/unified/queue_u2_finalize.py --wait-joint-pid 883611. ActualPID896659/session18355 confirms WAITING_ORIGINAL_JOINT_OWNER, pack_started=False, ready_for_u3=False. It waits real original completion, recomputes current full selection, packs actual frozen U2-B tensors, runs original-vocabulary retention and unmodified selected-core audits, then verifies every final gate against bound artifacts. Completed units resume by actual checkpoint receipt; reference fixtures and conditional U2-C cannot substitute for selected U2-B. No original owner was restarted, no additional formal optimizer launched and no U3 readiness claimed.

## U2 native image-batch admissibility probe — 2026-10-12

Previous goal turn made progress: selected-artifact finalization and bound real-audit/readiness entries were committed4102740, backed up and verified at GitHub592d097. Current worktree/project are correct/clean; original full TAO parent63615 is confirmed live through its existing blocking handle, and finalization waits original883611. No actual formal baseline/cache is ready. Read native inference_detector and fusion attention before considering future throughput: list-image API still loops singleton test_step, whereas the multimodal fusion uses a scalar max over the full batch. Probe actual FP32 native batch2 versus singleton on the two fixed original MOT17 frames, plus a second-image perturbation while keeping the first fixed. This is a small admissibility measurement, not a new formal audit or source/model change. Reject batching if candidate parity or first-frame independence fails; keep running jobs, candidate policy and original native singleton protocol unchanged.

Native image-batch probe10904 exit0 yields a genuine negative optimization result: same native preprocessing shape on two fixed MOT17 frames, but batch2 boxes/scores differ from singleton on both. Max same-rank score differences.000565775/.002606869; large coordinate differences1033.97/1865.55 reflect changed sorted proposal alignment and are not a matched-box localization error measurement. All labels remain equal. Actual six fusion stable_softmax_2d flags are False; the suspected scalar max is inactive, and changing only second-image pixels leaves the first batch output bit-exact in this probe. Reject native batch2 on candidate parity; no full OVMOT batch or semantic-feature integration is attempted, no TF32/precision tolerance or candidate policy is changed. Keep native singleton plus already proven fixed-text reuse. Original full TAO/perception/baseline/finalization jobs remain under their original blocking owners and gates; no tracking benchmark or U3 readiness is claimed.

## U2 upcoming perception-audit fixed-text reuse — 2026-10-12

Previous goal turn made progress: actual FP32 native image-batch parity failure was measured and committed24be26a/published64cb0ff; retain native singleton. Current project/worktree are correct/clean, and existing full TAO63615 blocking handle is confirmed live. Read current full TAO worker and bounded adapted-checkpoint evaluation caller. The already bit-exact fixed native text reuse can reduce repeated38-caption BERT work in upcoming adapted TAO/C-TAO audits too. Add an explicit optional CLI flag with unchanged default F0 behavior; only future adapted-checkpoint evaluator requests it. Keep identical singleton/native NMS, protocols, GT denominators, checkpoint/source fingerprints, resume rows and gates. Validate actual unchanged two-source smoke rows/recall counts with and without the flag before retaining this integration. Do not modify or restart the current loaded F0 workers.

Actual paired portability-worker37402 exit0 confirms optional text reuse: original two-source smoke metadata/rows identical,300 proposals each, TAO2base/1novel GT and C-TAO10base GT hit counts unchanged, one backbone/image, BERT38calls/38hits across two complete1203-class passes. Native/reuse elapsed11.545474/10.907659s are limited two-image cold/warm observations, not full throughput. Flag defaultFalse preserves F0 metadata/commands and resumable exact frame prefixes; only future adapted-checkpoint evaluation explicitly passes it to new workers. Keep this verified integration; all original full processes, protocols, GT denominators, weights, budgets, seeds and acceptance criteria remain unchanged. Actual adapted full audit and all tracking results remain unmeasured.

## U2 full F0 completion and native checkpoint load repair — 2026-10-12

Original blocking TAO audit63615 returned exit0/COMPLETE: all36,375 TAO validation and17,131 C-TAO original training frames measured. TAO base105,408/110,055=.957775658, novel2,777/3,057=.908406935; C-TAO base50,975/53,263=.957043351, novel inapplicable. Both OV portability streams pass; BDD/MOT17/MOT20 ordinary F0 gates still fail, so bounded U2-P is required. Its original owner11636 genuinely terminated at native MMEngine checkpoint loading before any optimization, because PyTorch's default weights_only=True rejects the original checkpoint's HistoryBuffer. Plan: retain original full measurements and exact eight-GPU replay/protocol; apply the existing U1 process-local TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD convention, verify the actual native loader on immutable U1R2, then resume the unfinished epoch and terminal downstream queues with updated owner PIDs. No metric, seed, budget or acceptance change.

Targeted regression94105 exit0: all six terminal shards exactly cover53,506 unique registered frames, every frame has one backbone call and300 native proposals, both source GT denominators and measurement totals match. The native MMEngine loader now loads the immutable SHA-verified actual U1R2 state_dict1,069 tensors plus HistoryBuffer/optimizer; full unchanged preflight is READY. No epoch checkpoint or first optimization update existed at failure. All four original downstream handles returned terminal HELD states; recover their owners only after this evidence. Keep the localized compatibility fix.

Recovery blocking supervisor23776 launched genuine new owners after all old handles terminated: perception1054069, ordinary1054070, MOT-only1054071, joint1054072, finalization1054074. Original failed states preserved in outputs/unified/u2/recovery_20261012/*.original; original training log retained. The single supervisor waits these exact processes; downstream unchanged gates prevent cache or tracking launch before a fully selected passing foundation. Full budget remains1,250 steps/epoch, at most2epochs, followed by6,000 MOT-only and eligible4,000 joint steps. Actual selected pack and U3 readiness remain unproven.

## U2-P epoch1 complete; adapted audit worker argument repair —2026-10-12

Recovery23776 returned genuine terminal exit1 after epoch1 completed all1,250 updates and saved epoch_1.pth (SHA1d9077ae603f40de0e955a11a88083f914f98faa967bc46cdd06bcdfb5129688). Actual checkpoint contract PASS compares265 immutable BERT/Swin0/1 tensors bit-exactly with original U1R2. First actionable evaluation failure: all ordinary workers exit2 before inference because the parent omits --measurements-dir, so their adapted-checkpoint parser sees the default F0 directory. The TAO parent has the same omission and would fail at its next stage. Plan: propagate the already explicitly supplied independent directory in both existing worker commands; verify actual adapted native smoke workers through both parent launchers, preserving guards and scopes; resume unfinished full evaluation from the completed epoch1 checkpoint, never rerun epoch1 optimization. All recovery downstream owners are terminal HELD, and will be restored behind the new perception owner.

Storage rebalance requested by user —2026-10-12: move only11 completed/CRC-extracted TAO archives from /data0/user (87.981GiB,49.54% of prior usage) and /data1/user (85.328GiB,48.72%) to matching /data3/user/TAO-Amodal/frames paths. Destination has789GiB free; extracted train/val/test image paths and scientific protocols remain unchanged. Two parallel volume transfers verify each destination against its existing download SHA before atomically replacing the original archive with a symlink; no source data is removed before verified copying. Plan recorded in outputs/unified/u2/storage_rebalance_20261012/plan.json.

User superseded archive relocation: original transfer69601 stopped with exit143; deleted all21 CRC-extracted completed TAO archives across allfour volumes (345.674GiB) plus cancelled relocation temporary copies (68.760GiB). New plan migrates approximately half of cleaned /data0/user and /data1/user image storage, using only test/validation directories; train directories remain in place. rsync validates transferred data, then a dry-run checks complete matching trees; Linux renameat2 exchange atomically replaces each original directory with a target symlink before removing its verified redundant original copy. Scientific image paths, original source annotations and all protocols stay valid. New explicit plan: outputs/unified/u2/storage_rebalance_20261012/image_migration_plan.json.

Post-cleanup image migration completed:95 directory units, /data0/user44.814GiB and /data1/user44.907GiB relocated to /data3/user, each49.999% of cleaned initial occupancy. rsync full-tree parity passed before atomic symlink exchange; all original image paths remain resolvable, only test/validation storage moved, train paths and experimental contracts unchanged. Final actual directory sizes and df recorded in outputs/unified/u2/storage_rebalance_20261012/image_migration_result.json.

Adapted launcher regression96599 exit0: through the actual ordinary and TAO parent launchers, each actual child receives the separate measurements directory and real epoch1 checkpoint SHA; one original legal KITTI frame and original two-source TAO/C-TAO smoke frames execute native inference successfully. Test-only smoke injection and collector do not produce formal measurements or weaken aggregate completeness; report is explicitly reference_only. Full recovery supervisor95952 resumes epoch1 evaluation from existing epoch_1.pth, new perception1089957/ordinary1089958/MOT-only1089959/joint1089960/finalization1089961 owners retain unchanged budgets and gates. All previous owners returned terminal, their failure states retained. Full U1 retention and adapted portability remain pending; no tracking or U3 claim.

User-requested inference sharding throughput plan —2026-10-12: full audits already use8 physical GPU shards, while native full1203-class TAO performs38 multimodal transformer passes per image. User asks to shorten runtime. Temporarily SIGSTOP exact evaluator1090197 and its existing TAO parent1238269/workers1238270..1238277, retaining all completed ordinary measurements and existing TAO exact frame prefixes; no original source inference re-started. Measure1/2/4 independent singleton processes on GPU1, real adapted epoch1 SHA1d9077a, identical12 reference frames spanning original TAO source domains, full vocabulary/native FP32/static text reuse and exact boxes/scores/classes parity. This is a bounded throughput probe, not a metric or scope change. Adopt higher concurrency only if actual throughput improves with unchanged outputs; otherwise retain the fastest measured concurrency and resume the existing owner. Probe shell60967; hold receipt outputs/unified/u2/perception_adaptation/throughput_probe_hold.json.

Same-GPU probe60967 completes: identical12 actual native full-vocabulary reference frames take158.996s with1 process,251.286s with2 and220.054s with4; all proposal boxes/scores/classes are bit-exact. Reject extra processes: single worker per GPU is fastest on this sample. Initial default-device memory counters were on GPU0 instead of model GPU1; omit those zero counters from published aggregate, retaining actual wall-clock/parity evidence.

Native metadata optimization85924 completes: explicit target CUDA binding plus one bulk integer-label copy for native label_names construction reduces the same12-frame sample to103.870s (observed1.531x), boxes/scores/classes bit-exact for all12, native full label_names identical on the first full-vocabulary frame; native methods restored. Actual model GPU1 peak allocation1,664,202,752bytes. This comparison includes CUDA binding and changed background load (seven formal audit workers resumed), so it is not an isolated causal label-copy benchmark or a full-run speed guarantee. Keep integer-only optimization provisionally; verify actual two-source native worker rows before resuming formal prefixes with it. Vocabulary1203, chunks32/38, NMS.8, Top300, precision/weights/protocol/GT denominators remain unchanged.

Actual bulk native worker63758 exit0: unchanged two-source smoke rows/GT hit counts and resume metadata are exact,300 candidates per frame, one backbone per frame,76 head calls/68,400 integer class labels; native/bulk CLI wall68.842/51.966s includes initialization and two frames, not formal throughput. After verified controlled interruption,95952 returned terminal exit1 (expected interrupted TAO command) and all four gated followers returned HELD without tracking launch. Every existing exact TAO prefix parsed successfully:440 completed frames preserved across initialized shards; shard1 had not yet written metadata or rows. All76,068 ordinary frames and epoch1 original1,250 updates retained. New supervisor57530 launches perception1313649/ordinary1313650/MOT-only1313651/joint1313652/finalization1313653. Full evaluation skips complete ordinary measurements and resumes unchanged eight TAO partitions with explicit GPU binding plus bulk integer metadata; labels alone temporarily use CPU for native name construction and return to the original device before NMS. Both flags preserve full1,203 vocabulary,38 native chunks, FP32, backbone counts, GT scope and all gates.

Actual epoch1 ordinary summary: BDD Top150.936574, Dance.999036, MOT17.878066, MOT20 Top150.819136/Top300.973281, original legal KITTI.946394. All five full ordinary streams are measured; MOT20 satisfies the registered Top300 crowded threshold but formal registration/selection still waits complete TAO and U1 retention. No selected foundation, candidate cache, tracking benchmark or U3 readiness is claimed.

### 2026-10-13 — U2 perception epoch 1 selected

- Completed the unchanged full TAO/CTAO portability audit: 36,375 / 17,131 images, Top300 base recall 0.9742855845 / 0.9729080225; TAO novel recall 0.9460255152 (2,892 / 3,057). Unmapped TAO GT retained separately (305 / 314 hits).
- Epoch 1 foundation `1d9077ae603f40de0e955a11a88083f914f98faa967bc46cdd06bcdfb5129688` passed all seven portability streams and U1 retention: generic V1/V2 0.9918663023 / 0.9672269380; expression Top100 V1/V2 1.0 / 0.9785374828. Fixed 500-pair language audit true/shuffled AP50 0.7301285093 / 0.4067468580. Generic scope 4,201 images; expression scope 7,160 representative query frames, 7,199 target references.
- Selection source receipts and their registered digests agree; retain epoch 1, no second adaptation epoch required. Existing recovery owner exited successfully and advanced the eight-GPU ordinary candidate cache automatically. Formal tracking scores and U3 readiness remain pending.
- Evidence: `outputs/unified/u2/perception_adaptation/eval_epoch_1/selection.json`, seven `measurements/*.json`, and generic/expression/counterfactual `result.json`.

## GroundJEV-MOT execution redirection and current handoff — 2026-10-18

The active task was re-scoped to `TRACKJEV_THREE_TASK_DETECTOR_ADAPTATION_AND_TRAINING_V1` under the continuing GroundJEV unified MOT/RMOT/OVMOT goal. The required project root for this checkout is `/data2/user/LocateMOT`; the DeepSeek Harness jobs that are still running use the separate research checkout `/data2/user/LocateMOT_OneJev_CVPR` and the model/runtime checkout `/data2/user/LocateMOT_GroundJEV`. This path split is recorded explicitly; no source from the live old checkout was copied into this U2 checkout.

Resource audit: GPU0–3 and GPU5 are occupied by another user's `/data2/OCD/TALON` jobs and were left untouched. GPU4 and GPU6 run the real three-task OneJev LoRA SFT; GPU7 hosts the protected GroundAnything decoder/feature services and the MOT20-05 export. There is no active pure MOT-only training process. The active SFT reads 43,533 rows (identity 21,168, RMOT 16,677, OVMOT 5,688; `native_G0` 21,168 and `GT_teacher` 22,365) and has a resumable FP32/Adam/RNG checkpoint at step 250 while the log has reached step 260/1,200. The protected MOT20-05 export has reached frame 1,558/1,658 in its resumed output; occasional decoder rejections are recorded as empty frames and do not terminate the run.

GroundAnything G1 was not idle: the existing evidence shows 200/200 projector steps completed and a G0/G1 comparison. The measured G1 recall is within about 1.2 points of G0 and is mostly lower, so it is retained as a negative detector-adaptation result and will not be relaunched on another user's GPU5 allocation. The next detector-related work is expression-conditioned RMOT, because the earlier RMOT native cache used a pedestrian-only request and failed its semantic quality gate.

Prepared without GPU inference: `/data2/user/LocateMOT_OneJev_CVPR/outputs/groundjev/full_data_v2/candidate_export_rmot_expression_v1/plan.json` and 24 per-query manifests cover 12 legal fit/calibration videos and 4,416 frames, exclude all held-out/evaluation videos, preserve the real expression text/files, and use `proposal_source=native_G0_expression_conditioned_pending`. The planner is `tools/groundjev/prepare_expression_rmot_pilot.py`; its manifest validation passed. Inference is intentionally deferred until the protected MOT20 export releases the shared decoder/feature service, so no existing long job is contended or restarted.

The active goal remains in progress. Next recovery points are: let the current MOT20 export and three-task SFT reach their existing checkpoints; run the expression-conditioned RMOT pilot from the prepared manifests; then evaluate the latest valid OneJev checkpoint on the independent held-out scope. No benchmark success or full-goal completion is claimed here.

The first expression-conditioned pilot attempt was stopped after the protected MOT20 owner was found to have exited at frame 1561 because its default 2 GiB native-feature budget was exceeded (the old prefix through frame 1560 remains intact and has no result receipt). The partial first RMOT query is also preserved as `*.failed_attempt`. The smallest recovery is now active: MOT20 resumes from that exact prefix in `mot20_MOT20-05_resumed_v2` with an explicit 4 GiB per-study budget (PID 3743898), while a single-worker RMOT queue (PID 3744099) waits for that new `result.json` before touching the shared decoder/feature endpoints. No result or protected source directory was overwritten.

## 2026-10-18 21:48 CST — continuation runtime audit

J1 v4 birth remains live after the step-300 checkpoint: torchrun PID 3029659 with ranks 3029872/3029873 on GPU4 and GPU6. Both ranks are still consuming compute (rank0 at full CPU and both model GPUs at 100% utilization), so the absence of a newer log line is an active long batch rather than evidence for a restart. The authoritative restore point remains `adapter-step300` plus `latest_training_state.pt`; no checkpoint or optimizer state was overwritten.

MOT20-05 `resumed_v2` is also live and writing the protected output: `tracks.mp4`, `predictions.txt`, `frame_candidates.jsonl`, `frame_decisions.jsonl`, and `latest_identity_memory.json` advanced after the previous audit. Its final `result.json` is still absent, so the expression-conditioned RMOT queue correctly remains `WAITING_MOT20_RELEASE` with zero model-inference queries. The expanded CPU-only J1 held-out gate remains alive at PID 3791075 and waits for the J1 final receipt plus MOT20/RMOT release; no held-out metric is claimed yet. No process was restarted or stopped.
## 2026-10-18 21:51 CST — MOT20 export advanced while J1 batch remained active

The protected MOT20-05 `resumed_v2` owner made real progress after the previous audit: `frame_decisions.jsonl` reached 12,201,850,168 bytes, `latest_identity_memory.json` reached 300,275,019 bytes, and `predictions.txt` reached 11,269,005 bytes at the latest snapshot. The final `result.json` is still absent, so the export is not yet released to RMOT. J1 still has only the step-300 receipt, but both distributed ranks remain live and one rank is actively computing; the current evidence does not justify a restart. The held-out gate and expression-conditioned RMOT queue remain correctly waiting on their registered receipts.
## 2026-10-18 — selective cache policy reaffirmed

Newly downloaded datasets do not require whole-dataset native feature/candidate caches. The formal J1 SFT already consumes the registered 43,533-row three-task course with media on demand and `feature_cache=false`. Cache/export is limited to registered evidence scopes: the protected MOT20-05 G0 long export, the legal expression-conditioned RMOT pilot manifests after the MOT20 release, and the explicitly registered held-out MOT/RMOT/OVMOT evaluation manifests (DT1203, Refer-KITTI validation scope, and the two TAO validation videos). Other downloaded train/validation data remains distributed across the data volumes and is read on demand; no full TAO or full Refer-KITTI cache is started. The older pedestrian-only RMOT cache remains a rejected semantic attempt and is not promoted or reused as expression-conditioned evidence.
## 2026-10-18 — G1 continuation decision and record boundary

G1 is a completed detector-side adaptation, not an active optimizer: the registered legal TRAIN mix spans MOT17, MOT20, DanceTrack, and Refer-KITTI, with 200/200 projector steps, a resumable FP32/Adam/RNG state, and a materialized `GroundAnything-DLM-TrackProjG1` serving directory. The short unchanged G0/G1 controls are valid but mixed/mostly negative (MOT4 and RMOT single8 regress; RMOT multi12 is slightly higher), so G1-200 remains a preserved baseline and is not silently extended or promoted. The larger downloaded corpora are available, but a G1-extended run would be a new experiment: it must use legal TRAIN data only, keep G0/G200/G1-200 checkpoints, and rebuild any downstream OneJev rows before claiming an effect. Current long-running priority remains the protected MOT20 G0 receipt and the real J1 three-task checkpoint/held-out gate; GroundAnything server processes are inference services, not G1 fine-tuning.

## 2026-10-18 22:35 CST — MOT20 G0 receipt released; expression RMOT pilot started

The protected MOT20-05 resumed export produced and passed its terminal receipt: `VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION`, all 1,658 frames processed, 1,560-frame exact resume prefix restored, 224 decoder-rejected/strict-invalid frames retained as empty official prediction frames, 2,314,870,593-byte native candidate archive, and 4,322.94 seconds end-to-end wall time. The owner process remained alive briefly after writing the receipt while flushing its final result output, then exited without modifying the completed artifacts. The expression-conditioned RMOT queue was released only after this receipt and is now running its legal 24-query pilot; the latest owner state reports 2 completed queries, model inference started, and no failure. The J1 held-out gate remains waiting for the final step-1200 training receipt.

## 2026-10-18 22:59 CST — expression RMOT pilot progress

The released expression-conditioned RMOT pilot remains healthy: owner status RUNNING_EXPRESSION_RMOT_PILOT, 5 of 24 legal query manifests completed, model inference active, and failed=null. The queue is still single-worker over the shared GroundAnything services; no held-out or validation videos have been admitted.

## 2026-10-18 23:38 CST — expression RMOT pilot passed midpoint

The expression-conditioned RMOT pilot remains healthy at 13 of 24 legal query manifests complete. The owner is still RUNNING_EXPRESSION_RMOT_PILOT with failed=null; no held-out or validation video has entered the queue, and no optimizer update is performed by this export.
## 2026-10-18 — GroundAnything G2-A native optimization gate passed

The isolated GroundAnything G2 line was initialized from `models/GroundAnything-DLM` after auditing the pinned official source at commit `8a3b43a1b42952e117960e85d63ff173bb9fba0a`. The task text referenced `e054ab329...`, but that revision is not the checkout's actual official HEAD; the actual pinned revision and source-file hashes are recorded in the G2 audit receipts. GPU3 and GPU5 were free at the one-time resource audit; J1 on GPU4/6, protected GroundAnything services on GPU7, and the active MOT20/RMOT owners were left untouched.

G2-A passed a real native DLM forward/backward/Adam-step gate on three supervision modes: dense MOT20, referring RMOT, and TAO-Amodal OVMOT. Vision and the original language weights remained frozen; the projector exposed 27,267,584 trainable parameters and Qwen3 q/k/v/o LoRA exposed 11,796,480 trainable parameters across 36 layers. The official mixed objective remains 0.5 MDM + 0.5 causal, with projector/LoRA learning rates 2e-6/7e-6 and AdamW. Losses were 1.24458, 1.39893, and 1.53041, with finite nonzero gradients in both projector and LoRA and a completed optimizer update. Peak allocation was 39,124,467,712 bytes on one 80-GB GPU. A narrow Transformers 5.17 compatibility shim for the removed `cache_position` keyword is recorded and covered by the gate. This is a preflight receipt only; formal ten-dataset G2 training has not started while dataset coverage and legal TRAIN-only row preparation are audited.



## 2026-10-19 — GroundAnything G2 coverage audit, corpus, and DDP smoke

The resolved TRAIN-only audit follows the distributed data layout and avoids held-out media. TAO-Amodal has all 18,274 TRAIN images available across /data0, /data1, /data2, and /data3. RMOT26 has all images referenced by its primary train annotation rows, but its download receipt still records 20 failed assets, so the source remains PARTIAL_DOWNLOADS. BURST has 500 TRAIN annotation sequences but no legal TRAIN image root in this checkout; it is excluded rather than substituting TAO test frames.

The G2 row builder emitted 199,053 validated official rows with absolute image paths and bytes=null: MOT17 15,948, MOT20 8,931, DanceTrack 41,796, SportsMOT 28,574, AnimalTrack 11,543, Refer-KITTI-V2 4,333, AerialMind 45,625 dense plus 23,733 referring, RMOT26 296 referring, and TAO-Amodal 18,274. The 2,246 rejected rows are retained in a separate JSONL with source and reason; no row path is missing, no official evaluation video appears, and all rows pass the official uniform row contract. Corpus SHA is b8df7365a1b1cda47489f8044d86c030be9ab17c48724b5b5620befb5c9eb76e.

A first two-rank DDP launch under system Python failed before model loading because that interpreter had no accelerate package. The failure is preserved as an environment issue. Re-running with the same project .venv-onejev-py310 used by G2-A passed one native optimizer step on physical GPU3/GPU5, including finite gradients, distributed synchronization, and atomic adapter-step000001/latest_training_state checkpoints. The cards were co-resident with pre-existing work and were not modified. Formal full-data training is now gated only by the registered long-run launch and later held-out evaluation; no G2 quality score is claimed yet.


## 2026-10-19 — G2 formal full-data run launched

After the G2-A native gate, ten-source coverage audit, row-contract/path audit, and one-step two-rank smoke all passed, formal G2 training launched with two DDP ranks on physical GPU3/GPU5 using `.venv-onejev-py310`, master port 29619, max_steps=1200, and checkpoint interval 50. The input manifest SHA remains b8df7365a1b1cda47489f8044d86c030be9ab17c48724b5b5620befb5c9eb76e. The run directory is `/data3/user/LocateMOT-runs/GroundAnythingG2/G2_PROJECTOR_QWEN3_LORA_FULLDATA/formal_g2_1200`; no held-out media or existing J1/MOT20/RMOT owner is touched. Formal completion and detector/tracking evaluation remain pending.


## 2026-10-19 — G2 formal length-gate failure and minimal recovery

The first formal two-rank run reached a valid step-1 checkpoint, then stopped at row `mot20_MOT20-05_001167_dense`: the official DLM collator measured 2,193 text tokens against the temporary 2,048 preflight bound. No data, metric, seed, or split changed; the failure receipt is retained under the formal run directory. The trainer default was corrected to the official 4,096 text / 8,192 joint limits. A targeted collator check passed the failing row at 2,193 tokens, and a two-rank resume regression from the valid step-1 checkpoint completed step 2 with finite loss 1.46757, finite gradient norm 6.09755, and an atomic step-2 checkpoint. Formal 1,200-step recovery resumes from the latest valid step-1 checkpoint with the corrected length contract.

## 2026-10-19 — G2 official memory route repair and long-row regression

The corrected 4,096/8,192 length run reached the first long MOT20 row but exhausted the remaining memory because the custom trainer still used the legacy padded language loop while GPU3/GPU5 each carried an unrelated 18.5-GiB resident workload. No metric or data path changed and the valid step-1 checkpoint remains the recovery point. The trainer now follows the released DLM general memory contract: padding-free packing, non-reentrant language gradient checkpointing at every packed layer, chunk linear kernel, fused LM-head when available, and no activation CPU offload.

This environment does not provide the optional `flash_attn` or `fla` modules. The trainer therefore selects GroundAnything's official `cudnn_sdpa` packed clean-attention backend and its official `checkpointed_chunk` LM-head backend, respectively; the objective remains the same 0.5 MDM + 0.5 causal loss. A two-rank regression with one 2,193-token MOT20 row per rank completed one optimizer step on physical GPU3/GPU5 with finite loss 2.0606513, finite gradient norm 5.1535301, and atomic checkpoint/final receipts. A prior one-row DDP fixture was interrupted after rank imbalance was identified; it did not alter the trainer or corpus. Formal recovery from step 1 is now allowed to proceed; quality results remain pending.

## 2026-10-19 — G2 formal recovery relaunched

The repaired formal G2 run was relaunched from the preserved step-1 checkpoint with the unchanged 199,053-row corpus (SHA b8df7365a1b1cda47489f8044d86c030be9ab17c48724b5b5620befb5c9eb76e), max_steps=1200, save interval 50, seed 20261019, and two DDP ranks on physical GPU3/GPU5 (master port 29633). The active command uses the packed/checkpointed route and the recorded cudnn_sdpa/checkpointed_chunk fallbacks. It is still running; no G2 quality metric or completion claim is made until the terminal receipt is written.

## 2026-10-19 — TrackJEV redirect state audit

The protected MOT20 G0 long export is terminal and complete (`VIDEO_INFERENCE_COMPLETE_WITH_REJECTED_PERCEPTION`, all 1,658 frames, exact resume prefix retained). The expression-conditioned RMOT pilot is terminal and complete for all 24 legal query manifests with `failed=null`; its proposals are recorded as native G0 expression-conditioned evidence. The actual MOT-only queue is terminal `STOPPED_BY_USER_RESEARCH_PIVOT` with `formal_track_training_launched=false`, so no abandoned MOT-only optimizer remains to be resumed.

The real OneJev three-task course is active at step 550/1200 from the registered 43,533-row manifest: identity 21,168, RMOT 16,677, and OVMOT 5,688 rows, with multimodal DDP on GPU4/GPU6, FP32/Adam/RNG checkpointing, and no feature cache. The J1 held-out gate remains `WAITING_J1_FINAL` and owns no GPU until step 1200. GroundAnything G1 is preserved separately: its apply receipt records a 200-step projector delta with five tensors replaced and byte-identical frozen vision/language weights; the GPU5 retry2 G0/G1 controls completed, while the measured G1 detector comparison is retained as a negative/mixed control rather than promoted. No held-out result or final TrackJEV claim is made before J1 reaches its terminal receipt.

## 2026-10-19 — authoritative redirect continuation audit

The protected MOT20-05 G0 long export is now terminal and complete: all 1,658
legal TRAIN frames are present, the exact 1,560-frame resume prefix is retained,
224 rejected/strict-invalid perception frames remain in the output, and runtime
GT/future-frame access is false. The evaluation-only TRAIN-scope receipt reports
HOTA 0.433441, DetA 0.638348, AssA 0.294655, and IDF1 0.441085; this is not an
independent held-out result. The expression-conditioned RMOT pilot also completed
all 24 legal query manifests with native G0 expression-conditioned proposals.

The registered MOT-only queue has no live process and remains
STOPPED_BY_USER_RESEARCH_PIVOT; no abandoned optimizer is resumed. The real J1
three-task OneJev run is live on GPU4/GPU6 at durable step700, using 43,533 rows
(identity 21,168, RMOT 16,677, OVMOT 5,688) with FP32/Adam/scheduler/RNG state.
Its final receipt and independent held-out metrics are still pending.

The clean G2 run remains active on GPU3/GPU5 at the latest observed step131/18227.
The registered fresh GPU5 G1 queue is alive and correctly waiting for the clean
G2 terminal receipt; it must not contend for GPU5 or duplicate the preserved
negative/mixed G1-200 control. The G2 bundle exporter and J1 held-out gate remain
dependency-gated. No unified quality improvement is claimed. The full evidence
map is in `docs/GROUNDJEV/EXECUTION_REDIRECT_AUDIT_20261019.md`.

## 2026-10-19 — live dependency progress revalidated

The registered clean G2 owner is still alive on GPU3/GPU5 and its event log has
advanced to optimizer step141/18227 with finite loss; no terminal receipt or
durable step-500 checkpoint exists, so it was not restarted or altered. The J1
three-task torchrun is also alive on GPU4/GPU6; its log has reached 720/1200,
while `adapter-step700` remains the latest durable FP32/Adam/RNG restore point.
The GPU5 G1 queue and clean-bundle exporter remain live dependency owners, not
failed jobs. No MOT-only process appeared, and no held-out evaluation was
started before the required terminal receipts.

## 2026-10-19 — live dependency progress revalidated (step 172 / 750)

One authoritative snapshot confirms the clean G2 torchrun is still alive on
GPU3/GPU5 at event step172/18227; its terminal receipt and durable step-500
checkpoint are absent. The active process still reports pre-v2
`samples_global=16`; the read-only global-accounting audit remains the only
corrected interpretation and no process was reloaded. J1 remains alive on
GPU4/GPU6 with durable `groundjev-full-fp32-adam-rng-v1` step750/1200 and
complete FP32/Adam/scheduler/RNG state; its final receipt is absent. The J1
held-out owner remains `WAITING_J1_FINAL`, the fresh GPU5 G1 owner remains
`WAITING_G2_CLEAN_SUCCESS`, and the clean bundle exporter remains
`WAITING_G2_AND_G1_TERMINAL`. No MOT-only process was restarted and no
optimizer, sampler, checkpoint, service, or owner state was changed. This is
execution evidence only; no quality claim is made.

2026-10-19 — Read-only G2 accounting audit generated at the active formal run directory without opening the training JSONL in write mode. The audit contains 175 valid pre-v2 events, zero malformed lines, preserves raw-log SHA `752c4ebb69de0ecf6f2fd0f77d068409934a800a333d095c1f676c61d4480df3`, and reports `raw_log_unchanged=true`. Its latest event is step175: logged `samples_global=16` and `effective_tokens_global=17079`, corrected for world size 2 to 32 samples and 34158 tokens under the registered global-batch-32 interpretation. This is an accounting correction only; no trainer process was restarted and no metric claim is made.


2026-10-19 — live dependency progress revalidated (J1 step800 / G2 step298). The real OneJev three-task torchrun remains alive on GPU4/GPU6 with the durable groundjev-full-fp32-adam-rng-v1 step800 state (260255305 bytes); no step-1200 terminal receipt or independent held-out result exists. The clean G2 torchrun remains alive on GPU3/GPU5 at event step298/18227 with finite loss 1.787952 and 17.3765 GiB latest peak memory; its pre-v2 raw event still logs samples_global=16, while the read-only accounting audit remains authoritative for the corrected global batch 32 interpretation. G1 GPU5, clean bundle export, and held-out evaluation owners remain dependency-gated. No training process, optimizer, sampler, checkpoint, or service was restarted or modified.


2026-10-19 — clean G2 crossed the registered step-300 runtime milestone. The active GPU3/GPU5 torchrun is still live at event step301/18227 with finite loss 1.813610, 16842 logged effective tokens, and 17.3718 GiB latest peak memory; no terminal receipt or durable step-500 checkpoint exists. J1 remains live on GPU4/GPU6 at durable step800/1200 with no final receipt. The existing G1 GPU5 queue, clean bundle exporter, and J1 held-out gate remain dependency owners; no process or checkpoint was restarted or altered.


2026-10-19 — read-only J1 stall diagnostic. Durable state remains step800/1200 and the final receipt is absent. The torchrun parent and both rank workers are alive; at the snapshot the parent was in do_select and both workers were sleeping in futex_wait_queue_me, with no failure/exception receipt. This is compatible with a distributed synchronization or long batch and is not evidence of a failed run, so no process, sampler, optimizer, or checkpoint was restarted.


2026-10-19 — J1 timing interpretation corrected from the read-only log: steps720–800 report 29.21–43.34 seconds per optimizer step, with finite losses/gradients and normal examples/tokens throughput. The absence of a newer durable state is therefore consistent with the registered long runtime (roughly hours for the remaining 400 steps), not a proven deadlock. The active process remains untouched.


2026-10-19 — full runtime audit after the blocking wait shows real progress. J1 log advanced through steps880/900/920 with finite loss and gradient values; the latest durable FP32/Adam/scheduler/RNG state is step900/1200 (step920 is logged but not yet a save boundary), with 68.917 GiB peak memory and no terminal receipt. Clean G2 advanced through events588/589/590 on GPU3/GPU5 with finite losses and about 17.377 GiB peak memory; no terminal receipt or step-500-only interpretation is used because the live event stream has passed it. G1 GPU5 and clean-bundle owners remain correctly waiting for G2/G1 terminal receipts, and all known processes remain alive. No active task was restarted or modified.


2026-10-19 — preserved read-only submission audit at docs/GROUNDJEV/LIVE_RUNTIME_AUDIT_20261019.json. It records G2 RUNNING at event630, J1 RUNNING with durable step900, G1 and G2-export owners waiting on registered dependencies, and the held-out gate in WAITING_J1_FINAL. The audit reports no non-finite/skip/truncation anomalies and does not mutate live runtime state.


2026-10-19 — runtime milestone update: J1 now has a durable step950/1200 FP32/Adam/scheduler/RNG state (log step940, finite loss and 43.86 s/step at that record); clean G2 has advanced to event step651 with finite loss and global accounting audit still applicable. No final receipt exists for either run. The held-out gate remains WAITING_J1_FINAL, G1 remains WAITING_G2_CLEAN_SUCCESS on GPU5, and the clean bundle exporter remains WAITING_G2_AND_G1_TERMINAL; no active task was changed.


2026-10-19 — J1 crossed the next durable boundary: latest log and FP32/Adam/scheduler/RNG state are both at step1000/1200 with finite training metrics; no final receipt exists yet. Clean G2 has advanced to event step786 with finite loss and the registered global-accounting audit unchanged. The held-out gate remains waiting on the registered J1 PID and final receipt; G1 and bundle owners remain dependency-gated. No active process or checkpoint was restarted or altered.


2026-10-19 — J1 durable step1050/1200 checkpoint is now present with FP32 masters, Adam moments, scheduler and all RNG state; the latest log is step1040 with finite metrics. Clean G2 reached event step900 with finite loss and global accounting fields. Neither terminal receipt exists; held-out, G1 and bundle owners remain dependency-gated, and no active task was restarted.


2026-10-19 — J1 durable step1100/1200 is complete with FP32/Adam/scheduler/RNG state and finite step-1100 metrics. Clean G2 crossed event1000 and is now at event1014 with finite loss. Neither terminal receipt exists; the registered held-out gate, G1 GPU5 queue, and clean bundle exporter remain dependency-gated. No live process or checkpoint was modified.


2026-10-19 — J1 durable step1150/1200 is present with the full FP32/Adam/scheduler/RNG recovery state; the latest logged step is 1140 and finite. Clean G2 advanced to event1167 with finite loss. Final receipts are still absent, so the registered held-out, G1 GPU5, and bundle owners remain waiting; no live task was restarted or changed.


2026-10-19 — J1 slow-batch diagnostic at log step1160: rank0 is sleeping in futex_wait_queue_me while rank1 is actively running (state R), with the torchrun parent in do_select. This confirms an active long batch/synchronization phase rather than a two-rank deadlock; the log mtime is recent and no failure receipt exists. No process or checkpoint was interrupted.


2026-10-19 — J1 training reached a valid terminal boundary. The log contains event=done with steps=1200 and 19,143 examples; latest_training_state.json is step1200 with FP32 masters, Adam moments, scheduler and all RNG state; adapter-step1200 contains adapter_model.safetensors plus adapter/processor/tokenizer configuration files. The torchrun exited after the done event without a standalone final_receipt.json, but the pre-existing CPU gate validated the step/adapter evidence, recorded j1_final and the preserved MOT20 G0 release, and advanced to RUNNING_HELDOUT_DT1203. No duplicate gate or training restart was launched.


2026-10-19 — J1 held-out gate advanced beyond the dependency waits: it has recorded j1_final at step1200, the preserved MOT20 G0 release, the completed expression-conditioned RMOT pilot, and a successful 186-tensor LoRA merge with zero unmatched adapter keys. It is now running the DT1203 held-out evaluation; result.json and official metrics are not yet present, so no held-out quality claim is made. No duplicate evaluator was launched.

2026-10-19 — held-out gate runtime milestone after the registered blocking wait. The gate owner advanced to RUNNING_HELDOUT_RMOT_MULTI12 on GPU4. DT1203 held-out inference and official TrackEval completed successfully with the registered 1203-frame scope; its metrics receipt is explicit and reports HOTA/DetA/AssA/IDF1=0, IDSW=0, FP=0, FN=4586, with rejected frames retained in the official denominator (this is an observed result, not a quality claim). The registered RMOT single8 clip also completed and its official receipt reports HOTA/DetA/AssA/IDF1=0, IDSW=0, FP=0, FN=5. RMOT multi12 inference produced a complete 12-frame result, but all frames were rejected by the strict duplicate-coordinate parser and its official evaluator remains pending; the gate process is still alive, so no terminal verdict is recorded. Clean G2 remains alive at event step1615/18227 with finite loss 1.210435 and about 17.37 GiB peak memory; no G2 receipt exists. No active job, checkpoint, seed, metric protocol, or evaluator was restarted or changed.

2026-10-19 — RMOT multi12 held-out stage completed. The gate recorded `rmot_multi12`; the 12-frame inference archive and official TrackEval receipt are both present. The registered short functional clip reports HOTA/DetA/AssA/IDF1=0, IDSW=0, FP=0, FN=30, with 30 released query GT boxes and all strict parser rejections retained in scope. The gate then entered RUNNING_HELDOUT_OVMOT_VIDEO4 on GPU4; OVMOT video4 inference is live and has not produced a terminal result or official metrics yet. Clean G2 remains independent and active. No active evaluator or training process was restarted.

2026-10-19 — OVMOT video4 live-progress validation. The held-out gate remains RUNNING_HELDOUT_OVMOT_VIDEO4 with no failure; the original evaluator process and ffmpeg encoder are alive. Its frame_decisions.jsonl has advanced to frame 1561 and the run log to frame 1591, with both files modified recently; result.json and official metrics are intentionally still absent while inference continues. Clean G2 has advanced to event step2136/18227 with finite loss 1.290262 and about 17.372 GiB peak memory. No evaluator, trainer, checkpoint, or protocol was restarted.

2026-10-19 — OVMOT video4 held-out evaluation completed. The gate recorded `ovmot_video4`; inference and the official TAO OVMOT TETA evaluator both completed for the registered 40-frame legal validation scope. The official receipt reports TETA/LocA/AssocA/ClsA and all recall/precision components as 0.0, 146 GT boxes, 0 predictions, and strict perception rejections retained in the denominator. This is the observed registered-scope result, not a quality claim. The gate advanced to RUNNING_HELDOUT_OVMOT_VIDEO63 on GPU4; its inference is live and has not produced a terminal receipt. No task or checkpoint was restarted.

2026-10-19 — OVMOT video63 live-progress validation. The final held-out gate remains RUNNING_HELDOUT_OVMOT_VIDEO63 with no failure; the evaluator and encoder are alive. Its frame_decisions.jsonl has advanced to source frame 436 and the run log to frame 465, with recent writes; result.json and official metrics remain pending during inference. Clean G2 has advanced to event step2635/18227 with finite loss 1.174764 and about 17.372 GiB peak memory. No task, checkpoint, or protocol was restarted.

2026-10-19 — OVMOT video63 live-progress validation (second milestone). The held-out gate remains RUNNING_HELDOUT_OVMOT_VIDEO63 with no failure; the original evaluator and ffmpeg encoder are alive. frame_decisions.jsonl has advanced to source frame 726 and the run log to frame 755, with writes about four minutes old; result.json and official metrics remain pending during inference. Clean G2 has advanced to event step2817/18227 with finite loss 1.297216 and about 17.370 GiB peak memory. No task or checkpoint was restarted.

2026-10-19 — independent held-out gate fully completed. The registered gate is now HELDOUT_ALL_REGISTERED_TASKS_COMPLETE with failed=null. Validated receipts exist for DT1203 (1203 frames; HOTA/DetA/AssA/IDF1=0, IDSW=0, FP=0, FN=4586), RMOT single8 (8 frames; HOTA/DetA/AssA/IDF1=0, IDSW=0, FP=0, FN=5), RMOT multi12 (12 frames; HOTA/DetA/AssA/IDF1=0, IDSW=0, FP=0, FN=30), OVMOT video4 (40 frames; all official TETA components=0, 146 GT boxes, 0 predictions), and OVMOT video63 (40 frames; all official TETA components=0, 160 GT boxes, 0 predictions). Every result uses the registered evaluator scope and retains perception rejections; no quality improvement is claimed. J1 step1200, the merged 3-task model, MOT20 G0, and RMOT pilot evidence remain preserved. Clean G2 is still running at event step3043/18227 with finite loss 1.032851 and about 17.370 GiB peak memory; GPU5 G1 and the G2 bundle exporter remain dependency-gated and untouched.

2026-10-19 — held-out zero-metric root-cause audit. The official evaluators are functioning and their receipts are protocol-faithful, but the J1 held-out outputs are empty. RMOT single8 and multi12 reject 8/8 and 12/12 registered frames respectively under the strict parser, with representative `parse_error=duplicate coordinate tuples are not canonical`; `perception_failure_policy=empty_and_record` then emits empty predictions. OVMOT video4 and video63 reject all 40/40 registered annotation frames under the same parser and their official receipts explicitly report predictions=0. DT1203 has nonempty detector candidates on 1201/1203 frames, but `outputs` is empty for all frames because the J1 decision head selects NO_MATCH for every candidate; the two rejected frames show duplicate coordinate/duplicate phrase errors. Thus the zero scores reflect parser rejection plus an over-conservative J1 association/calibration failure, not an evaluator bug. Original receipts remain immutable evidence; any parser/calibration repair must use a new, separately named experiment and must not overwrite this gate.
2026-10-19 — G2 Clean midterm checkpoint/evaluation and GPU7 release: the immutable strict recovery checkpoint at step3000 was validated (manifest SHA f682519a..., base model SHA 3cf09355...), extracted with the new read-only `export_g2_midterm_checkpoint.py`, merged to `G2_CLEAN_MIDTERM_STEP003000`, and reloaded successfully. A first short-control run exposed a stale 8101 G0 service whose scheduler was defunct and timed out at frame302; its failed directory is preserved. A fresh original G0 service on GPU6/8114 and the step3000 G2 service on GPU4/8113 then completed the same MOT17-02-FRCNN 4-frame, Refer-KITTI 0007 single8, and multi12 scopes with the official_scoring parser, tracking_dense decoder policy, Simple association, and TrackEval commit 12c8791b. G0→G2 HOTA/IDF1: MOT 0.750687/0.839695→0.861959/0.894737; RMOT single 0.303853/0.400000→0.472702/0.555556; RMOT multi 0.226643/0.217391→0.448010/0.387097. Multi IDSW increased 0→11; this is short functional evidence, not a final/held-out claim. GPU7 audit found MOT20-05 G0 complete at 1658 frames, expression RMOT 24/24 complete, and no active consumer/supervisor; targeted SIGTERM stopped 8102/8109/8112, ports closed, and GPU7 memory fell 43518→0 MiB. Preserve original G0/J1 zero-score results and all failed/stale-service evidence; G2 main training remains active on GPU3/5.

2026-10-19 — J1 parser/NO_MATCH repair live evidence snapshot. The original all-zero held-out gate remains immutable. The repaired RMOT smoke replays are nonzero (single8 HOTA=0.379221, IDF1=0.666667; multi12 HOTA=0.159866, IDF1=0.146341). The repaired full DT1203 run completed 1203 frames and official TrackEval with HOTA=0.158562, DetA=0.197692, AssA=0.127243, IDF1=0.128277, IDSW=0, FP=110, FN=3560. The subsequent OVMOT replay is still running and records malformed GroundAnything decoder wire payloads (missing coordinate/box tokens, duplicate coordinates/phrases, and an invalid bbox), producing empty candidates on affected frames with a 250–281 second slow path. These errors are recorded rather than repaired by inventing boxes; a separately named decoder/schema replay is required. YesTrack has no active real process in this snapshot. The exact report and JSON are committed as `docs/GROUNDJEV/J1_ZERO_METRIC_REPAIR_REPORT_20261019.*` and `docs/GROUNDJEV/LIVE_EXPERIMENT_METRICS_20261019.*`; no model weights, videos, or caches are committed.

2026-10-19 — P0 TrackJEV dependency repair and independent verification. The real
implementation of `trackjev/__init__.py`, decision schema, candidate competition,
global assignment, losses, model, tracker decision, `global_identity_runtime.py`,
and `global_identity_core.py` was published to
`research/yestrack-onejev-verification-20261019` at commit `9e09ff9`; current
parser/native adapter/run_video files were included. A detached clean worktree
was frozen at local verification commit `b153174`. `compileall`, all TrackJEV
imports, decision/model/adapter self-tests, and a real OneJev checkpoint load
passed. The native smoke completed on real MOT17-02 frame 2 with seven
GT-derived candidates, global assignment and memory updates in 70.34 seconds;
it is plumbing validation only, not a tracking metric.

2026-10-19 — G2 evaluator identity repair. `run_g2_native_eval.py` now labels
adapted outputs `G2-CLEAN-V2`; the historical `G2-PL` identity is not reused.
The fix was pushed to `codex/groundjev-4gpu-20261013` at `9cdb7a4`.

2026-10-19 — verified OVMOT repair boundary. J1 repaired DT1203 remains
HOTA=0.1585619177, DetA=0.1976921614, AssA=0.1272432780, IDF1=0.1282768263,
IDSW=0, FP=110, FN=3560. OVMOT video4 completed 40 registered TAO frames with
40/40 strict perception rejections, 146 GT boxes, 0 predictions, all official
TETA components zero, and end-to-end FPS 0.0037955. Frame latency was
239.58--288.17 seconds due malformed decoder wire payloads. Video63 was not
started after this verified boundary so invalid output would not consume GPU
indefinitely; no video63 metric is claimed. The report and live JSON were
updated without changing model weights or the immutable original gate.

2026-10-19 — G2 live accounting update. The protected GPU3/GPU5 Clean V2
training remains running at event step 4889/18227, latest loss 1.4810857,
peak memory 17.3716 GiB. The recent 500-event mean is 14.51 seconds per step,
which gives a linear remaining estimate of 53.76 hours. No trainer, checkpoint,
optimizer, sampler, RNG, or world size was changed.

2026-10-19 — G2 short-control TRAIN overlap audit completed read-only over all
583,251 Clean V2 rows. The 24 control frames were hashed individually and
matched by normalized image path. All four MOT17-02-FRCNN frames 302--305 have
exact TRAIN rows (the video also has 602 TRAIN rows in the manifest), so that
MOT result is TRAIN-scope/non-independent. The eight RMOT single and twelve
RMOT multi Refer-KITTI 0007 frames have no exact TRAIN path matches in this
manifest. Receipt: `docs/GROUNDJEV/G2_SHORT_CONTROL_TRAIN_OVERLAP_AUDIT_20261019.{json,md}`.
No training row or manifest was changed.

2026-10-19 — J1 strict DT1203 A–E replay ablation completed on the step1200
merged checkpoint (model SHA 5cc64d1b...). All association runs use recorded
candidate/native-feature archives and no GT at runtime. A original parser +
original NO_MATCH: HOTA/IDF1=0/0, 9,755 candidates, 9,729 NO_MATCH, zero
outputs. B repaired parser only: candidate count rises to 13,853 but learned
NO_MATCH still suppresses every output, HOTA/IDF1=0/0. The original runtime
already applied the legal MOT vocabulary fallback on 972 records, so C is
explicitly non-separable from A and is reported as the same immutable receipt.
D only `NO_MATCH=only_if_omitted` on original candidates yields HOTA=0.162326,
DetA=0.202661, AssA=0.130116, IDF1=0.132145, IDSW=0, FP=69, FN=3520, with
1 NEW/1133 ACTIVE/1 REACTIVATE/1 LOST events. E full repair yields
HOTA=0.158562, DetA=0.197692, AssA=0.127243, IDF1=0.128277, IDSW=0, FP=110,
FN=3560, with 1 NEW/1135 ACTIVE events. No threshold, seed, checkpoint or
candidate was changed within each factor; parser repair is not claimed as an
identity improvement. Receipt: `docs/GROUNDJEV/J1_DT1203_A_E_STRICT_ABLATION_20261019.{json,md}`.

2026-10-20 — AerialMind full-release acquisition and audit. The official
Hugging Face release was downloaded under `data/downloads/AerialMind` and
passed CRC tests for `expression.zip` (50,089,761 bytes),
`labels_with_ids.zip` (53,789,787 bytes), and `image_02.zip`
(24,425,047,497 bytes). The complete release was extracted to
`data/AerialMind`; the audit reports 93 videos, 24,617 expression JSON files,
45,630 image frames, 45,612 label frames, 8,479,541 expression-frame
references, and 46,135,642 target references. All referenced image and label
files exist, both official 17-video VisDrone and 13-video UAVDT test
partitions are present, and no JSON/label format errors were found. The 18
image frames without standalone label files are not referenced by an
expression; persistent target IDs absent from individual frames are treated
as normal occlusion/out-of-view state, not fabricated GT. Receipt:
`results/audits/aerialmind_dataset_audit.json`.

2026-10-20 — AerialMind manifest/evaluator adaptation. The runtime now uses
the official expression filename as query text, while retaining the existing
KITTI `sentence` behavior by configuration. Evaluation manifests record
`xywh_topleft` for VisDrone, `cxcywh` for UAVDT, and zero frame offset because
AerialMind filenames are 1-based. Predictions and GT were both verified on
the same frame convention. A direct TrackEval adapter was added to supply
explicit sequence lengths and avoid the pinned TempRMOT MOT wrapper's stale
hard-coded KITTI image path. The Transformers 5 tokenizer API difference was
bridged narrowly with a `batch_encode_plus` compatibility alias; the source
model remains frozen and checkpoint loading still reports zero missing or
unexpected model keys.

2026-10-20 — complete-data AerialMind single-card experiments on physical
GPU 2 (A800 80GB, 0 compute processes, CUDA UUID
`GPU-c2484ded-5f87-237c-bbf0-d23fcb74fdb2`). Two complete episodes were run:
VisDrone `uav0000009_03358_v` (219 frames) and UAVDT `M0203` (1,007 frames),
1,226 frame queries total, with no GT or future frames available to the
runner. The formal QCS suite (`FROZEN`, `PROBE_CACHE`, `QCS_TTA`, `NO_STATE`)
completed at 9.70 FPS shared-source throughput and 3.21 GB peak allocated
memory, but all methods selected zero rows and HOTA/DetA/AssA/IDF1 were 0.
The diagnostic `referring_threshold=0.2` run also selected zero rows. A
separate all-thresholds-zero candidate lower bound emitted 1,775 rows and
obtained combined HOTA 2.7327, DetA 1.0171, AssA 7.3499, IDF1 0.0; QCS was
identical to FROZEN with zero optimizer steps. The lower-bound per-episode
result was M0203 HOTA 4.4524 / DetA 2.6987 / AssA 7.3499 and VisDrone HOTA
0.0. These results show the data/GT/evaluator path is functional but the
Refer-KITTI-V2 checkpoint is not a viable AerialMind source model: its
absolute referring scores collapse on VisDrone and its first UAVDT frame has
no valid tracked candidates. Changing only the TTA residual structure cannot
repair missing source detector/tracker calibration; a target-domain source
training checkpoint or a separately trained domain-robust candidate source is
required before claiming AerialMind QCS gains. The full official AerialMind
training claim is intentionally not started on one card because the official
repository reports about 110 hours on eight GPUs; the supervised training
path remains the next experiment, with the current zero-shot runs preserved
as negative controls.

2026-10-20 — AerialMind relative-rank structure change and complete-episode
retest. The absolute referring gate was insufficient under the measured
target-domain calibration shift, so the runner now has an explicit
`relative_margin` mode: after admitting valid frozen tracks with zero
objectness/area gates, it keeps the set of candidates within a fixed 0.75
logit margin of the best current-frame referring score and caps the set at 16;
it never reads GT, future frames, or forces a top-1 decision. The original
absolute mode is unchanged and a CPU unit test covers both paths. On the same
complete VisDrone episode (219 frames) and UAVDT episode (1,007 frames), the
relative run completed all 1,226 queries at 9.35 shared-source FPS and 3.21 GB
peak allocated memory, emitted 1,344 rows, and sealed TrackEval for all four
methods. Combined FROZEN/PROBE_CACHE/QCS_TTA/NO_STATE each reported HOTA 3.0240,
DetA 1.2719, AssA 7.1964, DetRe 5.0987, DetPr 1.4568, and IDF1 0.0; the
previous all-thresholds-zero lower bound was HOTA 2.7327 / DetA 1.0171 /
AssA 7.3499 / IDF1 0.0. QCS still made zero optimizer updates and all four
methods were identical, so the change recovers candidate visibility but does
not establish referring adaptation or a useful source-free AerialMind result.
The complete official 30-video test manifest was also generated with 5,125,243
runtime rows and 5,125,243 evaluation rows; runtime contains no GT fields.
Receipts: `results/runs/aerialmind_two_episode_relative_rank/`,
`results/manifests/aerialmind_test_{runtime,evaluation}_manifest.jsonl`, and
`results/manifests/aerialmind_test_manifest_metadata.json`.

2026-10-20 — AerialMind supervised-training boundary audit. The official
`refer-uav.train` list maps 27,044 image rows to 63 training videos and 16,216
expression files; every listed image and label exists, every training video
has an expression directory, all query JSONs are valid, and the training
videos have zero overlap with the 30-video official test partition. Receipt:
`results/audits/aerialmind_training_split_audit.json`. This makes a future
target-domain supervised checkpoint path reproducible, but it is deliberately
not used by the source-free QCS experiments; the official repository's roughly
110-hour/8-GPU training claim is not replaced by a one-card partial run.

2026-10-20 — Refer-BDD complete-data audit and manifest preparation. The
official EchoTrack seqmap and Google Drive archives were downloaded to
`/data2/user/LocateMOT/data/downloads`, checksum/unzip validated, and the
complete selected split was extracted under `data/ReferBDD`. The audit found
8 videos, 673 expressions, 1,619 images, 1,610 label files, 74,924
expression-frame references, and 216,613 target-object references. A
label-free runtime manifest and a separate evaluator-only manifest were
created; no raw data or checkpoint is intended for publication.

2026-10-20 — Refer-KITTI-V2 → Refer-BDD source-free online experiment.
Hypothesis: frozen-query probes plus external causal residual adaptation could
recover source referring decisions under BDD domain shift without target
labels. The full six-method suite (`FROZEN`, `ENTROPY_HEAD`, `PROBE_ST`,
`PROBE_CACHE`, `QCS_TTA`, `NO_STATE`) ran on physical GPU 6 using the complete
673-episode/136,188-frame-query manifest. The runner used no GT, future frame,
expression label map, or evaluator; GT was materialized only after prediction
completion. The run completed in 32,513 seconds at 4.191 shared-source FPS,
with 8.44 GB peak allocated memory.

2026-10-20 — Refer-BDD experiment diagnosis and retained result. The first
smoke exposed an environment ABI mismatch and an evaluator +1 frame-offset
mistake; both were fixed and replayed before the full run. The full sealed
result is complete for all 8 videos and 673 expressions. FROZEN obtained
HOTA/IDF1 2.9788/0.89107. ENTROPY_HEAD made 31,656 real online optimizer
updates and degraded to 1.5842/0.28707. Probe/QCS/NO_STATE made 17,616 shared
probe calls but zero supported optimizer updates and remained identical to
FROZEN. Offline candidate analysis found approximately 26.4% IoU-0.5 recall
before gates, 19.8% after objectness/area, and 0.7% after the fixed 0.4
referring gate; source-score p95 was 0.3088. This is retained as a valid
negative cross-domain result; no BDD supervised training, threshold tuning,
GT-assisted structure change, or zero-gate result was substituted.

2026-10-20 — Result-accounting repair. The runner was corrected to accumulate
optimizer/probe counters across episode resets. The existing sealed output was
repaired transparently from `update_trace.jsonl`; the previous last-episode
values (50 entropy steps and 36 probe calls) are preserved in the metadata's
`counter_correction` field, while the published summaries report the complete
suite totals. The BDD asset audit, diagnostics, metrics CSVs, and final report
are kept; unrelated GroundJEV worktree changes were not staged or overwritten.

2026-10-20 — TTA-RMOT task boundary and ShiftRMOT-Bench-v1 implementation. This
is a separate repository/task from LocateMOT; all changes in this entry are
under `/data2/user/TTA-RMOT`. Hypothesis: compact paired visual/language
conditions can test source-free RMOT adaptation without storing duplicate
transformed images or exposing GT to the online runner. The complete
Refer-KITTI-V2 official test videos (`0005`, `0011`, `0013`, `0019`) are audited
before model execution; `0005` is a dev label and the remaining videos carry
an explicit historical-overlap warning rather than an unseen-split claim.
Images are symlinked to the external data2 asset store, visual shifts are
deterministic and geometry-preserving, and language rewrites are restricted to
audited directional phrasing. The runner/evaluator now expand compact rows at
their respective boundaries. The protocol labels for TDA/SAR/TPT/TCR are
explicitly marked adapted/inspired where the open-source method cannot be
reproduced exactly on TempRMOT's architecture; no result is retained until the
manifest audit and a paired Frozen/TTA run complete.

2026-10-21 — ShiftRMOT absolute-gate pilot. Four representative expressions,
45 A0/A1/A2/A3 conditions, and 9,915 frame queries completed on one A800
(`GPU-0322c0e7-ffbf-e068-5bdd-587b8216a9d9`). Frozen HOTA/DetA/AssA/IDF1 was
0.31255/0.090851/1.2371/0. The absolute gate emitted only 702 rows, all from
the 0019 representative episode; SAR_VISUAL_INSPIRED emitted 84 rows after
2,333 entropy updates and scored zero. TDA_RMOT, TPT_LANGUAGE_INSPIRED, and
TCR_FUSION_INSPIRED matched Frozen; probe calls occurred but no supported
optimizer updates occurred. This result is retained as a negative gate
calibration observation.

2026-10-21 — ShiftRMOT relative-margin pilot and structure decision. The same
45 conditions and 9,915 frame queries were rerun with objectness/area
candidate visibility followed by a fixed 0.75 current-frame referring-logit
margin and a 16-candidate cap, without top-1 forcing. On GPU UUID
`GPU-883c8be3-25b0-3d19-6fa8-4c2e335e67d2`, Frozen emitted 28,064 rows and
obtained HOTA/DetA/AssA/IDF1 2.9113/2.3094/4.2506/0.012108. The structure
change is kept because it recovered measurable candidates; it is not claimed
as a TTA gain. TDA/TPT/TCR remained identical to Frozen with zero updates;
SAR made 2,333 updates but decreased to HOTA 2.8306 and AssA 4.0007. Both
sealed evaluator outputs and the condition-level tables are retained under
`results/shift_rmot_pilot/`; the final report records that this is a
representative pilot rather than an all-4,540-condition claim.

2026-10-21 — ShiftRMOT-Bench-v2 complete Refer-KITTI-V2 data/protocol build.
The v1 audit confirmed that the runtime frame list was derived from expression
label keys and that visual seeds depended on expression/condition IDs. The
resulting risk was false negative frames during causal tracking and unequal
pixels between A1/A3 paired conditions. The v2 builder now inventories all 21
videos and 9,758 expressions, records the official train-list and seqmap
SHA256 values, keeps 8,008 source frames as external symlink targets, and
creates controlled conditions for all 862 official test expressions. Runtime
rows use every source-video frame in order; evaluator rows use only the
expression label keys after prediction. Visual seeds are independent of
expression and language and follow the registered video/frame/shift/severity/
global-seed protocol. Natural same-label-map language variants cover 825
expressions twice, 34 once, and 3 not at all; this gap is disclosed rather
than filled with unsafe rewrites. The complete manifest has A0/A1/A2/A3
counts 862/6,896/1,684/13,472. Pairing, no-GT runtime, symlink, target-set,
and time-axis audits passed; the audit status is
`PASS_WITH_DISCLOSED_LANGUAGE_COVERAGE_GAP`. Receipts:
`data/ShiftRMOT-v2/`, `results/audits/shift_rmot_v2_audit.json`,
`scripts/build_shift_rmot_v2.py`, and `scripts/audit_shift_rmot_v2.py`.

2026-10-21 — Independent v2 TTA implementation and source-training launch.
The previous v1 protocol labels were not reused as aliases. `TDA_RMOT` is now
a bounded non-parametric feature/reliability cache; visual, language, fusion,
and all pairwise/full combinations use separate zero-initialized residual
adapters over frozen TempRMOT snapshots. The language arm consumes the frozen
256-D RoBERTa-to-TempRMOT query projection, and the fusion arm includes an
explicit feature-query cross product and set-separation loss. Synthetic CPU
regression passed for all seven trainable arms with source-equivalent initial
logits, nonzero gradient, and parameter movement; TDA cache hits/updates also
passed. A wrapper sampling-length bug was fixed so inference follows the
official `hist_len=5` source-training configuration instead of the historical
hard-coded four-frame setting. The full 60-epoch official Refer-KITTI-V2
TempRMOT source run was launched on the selected free GPU 7; completion is
pending the independent checkpoint/optimizer/epoch audit. No full metric or
reproduction claim is made yet.

2026-10-21 — Official label-coordinate audit correction before metrics. A
cross-check against the pinned TempRMOT `datasets/refer_kitti.py` loader found
that Refer-KITTI KITTI label files are normalized top-left `x1,y1,w,h`, not
`cx,cy,w,h`. The first v2 manifest metadata had recorded the wrong convention;
it was corrected to `xywh_topleft`, the full manifest was regenerated, and the
audit was rerun successfully with the official time-axis, pairing, symlink,
and no-GT checks. No metric or training result based on the stale metadata is
retained or reported.

2026-10-21 — GPU7 source-training launch failure and controlled retry. The
first full TempRMOT Refer-KITTI-V2 launch was recorded with
`CUDA_VISIBLE_DEVICES=7`, but its launcher disappeared before producing any
training log, checkpoint, exit status, or end timestamp. A one-time GPU
diagnostic found three unrelated compute processes on physical GPU7, so the
card choice was not suitable for a clean single-card run; the exact kill
signal is not recoverable from the preserved files. No epoch, metric, or
completion claim is accepted from this attempt. The unchanged 60-epoch
official protocol is being retried under a new run root on physical GPU1,
which had one external process and substantially more available memory at
selection time. Receipt:
`results/source_training/temprmot_refer_kitti_v2_gpu7_launch_failure.json`.

2026-10-21 — GPU1 retry reached model initialization but failed on the first
training batch. The pinned TempRMOT code calls
`RobertaTokenizerFast.batch_encode_plus`, while the shared runtime provides
`transformers==5.2.0` without that legacy alias. The run exited with status 1
and produced no checkpoint or epoch result; this is recorded separately from
the GPU7 launch failure. A project-local `sitecustomize.py` compatibility
alias was added, and a local tokenizer regression passed before resuming the
unchanged training protocol. Receipt:
`results/source_training/temprmot_refer_kitti_v2_gpu1_transformers5_failure.json`.

2026-10-21 — Compatibility-resumed source training progress snapshot. The
third isolated run root
`/data3/user/TTA-RMOT-runs/temprmot_refer_kitti_v2_full_retry_gpu1_compat`
passed initialization and was executing the unchanged official protocol on
physical GPU1. At the one-time snapshot it had reached epoch 0, iteration
950/5168, with average loss 9.2707 and approximately 5.3 GiB peak memory;
the logger reported 08:02:08 remaining for that epoch. This is not a final
metric or completion claim. Receipt:
`results/source_training/temprmot_refer_kitti_v2_full_retry_gpu1_compat_progress_snapshot.json`.

2026-10-21 — Official TempRMOT Refer-KITTI-V2 checkpoint recovered for a
separate fast comparison line. The stale README link was corrected by reading
the official `zyn213/TempRMOT` v1.0 release metadata; `checkpoint_rk2.pth`
downloaded successfully to external asset storage, has 1,096,316,824 bytes,
SHA256
`4953bd2b2966f752b7d5dced94216f0ddacd04158b1c85c3fc3534a333e9581b`, and
loads as epoch 49 with 969 model keys. It is being used only for an official
Frozen/source-free-TTA comparison and is not substituted for the independent
no-leakage source retraining protocol. Receipt:
`results/source_training/temprmot_refer_kitti_v2_official_checkpoint_provenance.json`.

2026-10-21 — First released-checkpoint inference launch failed during model
construction because the new single-GPU wrapper omitted the official test
sampler arguments; the pinned source calls `max(args.sampler_lengths)` and
therefore raised on `None`. No prediction or metric was produced. The wrapper
is corrected to match the pinned `configs/temp_rmot_test.sh` settings
(`sampler_steps=50,90,150`, `sampler_lengths=2,3,4,5`, `hist_len=8`, and
`extra_track_attn`), and the retry will use a separate output root. Receipt:
`results/source_training/temprmot_official_inference_gpu3_sampler_args_failure.json`.

2026-10-21 — The corrected official-checkpoint inference reached the first
 297-frame sequence but failed on its first text-encoding call because the
 wrapper's runtime path insertion did not trigger Python's startup
 `sitecustomize` import. The same Transformers 5.2 compatibility alias works
 in the training launcher, which sets `PYTHONPATH` before interpreter startup.
 The wrapper now explicitly imports `sitecustomize` after path setup; no
 complete prediction or metric is accepted from this run. Receipt:
`results/source_training/temprmot_official_inference_gpu3_testcfg_transformers5_failure.json`.

2026-10-21 — A targeted strict-load regression exposed a second official
checkpoint issue. The V2 test shell enables `extra_track_attn`, but the
released RK2 state dict has no `update_attn` or `norm4` tensors; strict loading
reports 36 missing decoder tensors. The permissive official loader would leave
these layers randomly initialized, so the long GPU3 run was stopped before
accepting metrics. The checkpoint-compatible fast line now uses the source
training architecture (`extra_track_attn=false`, `hist_len=5`, sampler lengths
`5,5,5,5`). Receipts:
`results/source_training/temprmot_official_inference_gpu3_extra_track_failure.json`
and `results/source_training/temprmot_tta_release_smoke_gpu7_extra_track_failure.json`.

2026-10-21 — After disabling the untrained extra decoder attention, the
checkpoint-compatible independent runner loaded the official RK2 weight with
zero missing/unexpected model keys and completed a 10-frame A0 regression on
GPU7. The runner recorded `gt_used_by_runner=false`,
`evaluation_manifest_opened=false`, and `future_frames_used=false`; the
post-hoc evaluator also completed on the one-expression smoke output. This is
only a compatibility/evaluator regression, not a full-dataset metric. Receipt:
`/data3/user/TTA-RMOT-runs/shift_rmot_v2_official_release_smoke_gpu7_compat2`.

2026-10-21 — Full ShiftRMOT-v2 source-free online TTA was launched from the
checkpoint-compatible released TempRMOT weight. The 22,914-condition runtime
manifest is partitioned by deterministic condition index across four
independent shards on physical GPUs 3, 5, 6, and 4; shard outputs are merged
only after all predictions finish, then evaluated with the post-hoc evaluator.
Each runner is GT-free (`gt_used_by_runner=false`, no evaluation manifest and
no future frames). This is an explicitly labeled released-weight comparison
line: checkpoint provenance is external and cannot by itself certify
no-leakage, so the separate source-retraining line remains required.
Launch receipt:
`results/tta/shift_rmot_v2_release_full_sharded_launch.json`.

2026-10-21 — Revalidated the requested Refer-KITTI-V2 → Refer-BDD line. The
existing sealed result covers all 8 official videos, 673 expressions and
136,188 frame-query rows for FROZEN, ENTROPY_HEAD, PROBE_ST, PROBE_CACHE,
QCS_TTA and NO_STATE. The Refer-BDD assets, archives and manifests were
copied from the old shared checkout into TTA-RMOT-owned `/data2` paths; the
asset audit still passes with identical archive/seqmap hashes, and the copied
prediction/evaluation artifacts are byte-identical. The runtime manifest has
no GT fields and the historical run metadata records no GT, future-frame or
evaluation-manifest access. This establishes protocol isolation, but the
publisher's complete training history for the external RK2 checkpoint cannot
be independently audited; therefore this remains a released-weight result,
not an absolute no-leakage certificate. Receipt:
`results/audits/refer_bdd_pretrained_leakage_audit.json`.

2026-10-22 — Resource-cap correction. A live process audit showed that the
TTA-RMOT source-training line (GPU1), released-checkpoint official inference
(GPU7), and four auxiliary ShiftRMOT-v2 shards (GPUs 3/4/5/6) would exceed
the user-imposed maximum of four physical GPUs. The four auxiliary shards
were interrupted through their existing handles; their partial traces remain
for failure/progress provenance but are not merged or scored. The source
training and official inference lines were left running; no new TTA-RMOT GPU
job was started after the correction. Receipt:
`results/tta/shift_rmot_v2_release_full_sharded_launch.json`.

2026-10-22 — Task-boundary correction. A FlexHook asset/inference detour was
identified as unrelated to the requested TTA-RMOT protocol
(TempRMOT Refer-KITTI-V2 -> Refer-BDD). Its single-card run was interrupted
before completion, produced no accepted metric, and its scripts/receipt were
removed from the main experiment line. The valid Refer-BDD results and the
source-training/inference evidence remain unchanged.

2026-10-22 — Explicit FlexHook task resumption in an isolated TTA-RMOT
worktree. The user clarified that this is the TTA-RMOT/FlexHook task and not
LocateMOT or GroundJEV. Work therefore continued at
`/data2/user/FlexHook-TTA` on branch
`research/flexhook-generalizable-plugin-tta`, leaving the original
`/data2/user/TTA-RMOT` dirty files untouched. The fixed vendor commit is
`bd1acc38634b28525d54dc6e0fcb38335f0029f9`.

2026-10-22 — User-requested process cleanup. The obsolete long V2 source
training chain (PIDs `2777858`, `2777863`, `2777864`, `2778107`) was stopped
with TERM, preserving its checkpoints and logs. The separate old official
TempRMOT inference PID `3105941` was also explicitly stopped after confirming
it was not the FlexHook run. No unrelated GPU process was touched.

2026-10-22 — Official FlexHook Refer-KITTI-V2 reproduction completed. A
single-card GPU3 trial reached batch 280/3722 and was stopped for runtime; no
metric from it is accepted. The official two-process protocol then completed
on physical GPUs 1 and 4 across the four test videos and 862 expressions.
TrackEval returned HOTA 42.526, DetA 30.629, AssA 59.186, with 750,834
detections and 30,871 predicted IDs. HOTA matches the official README. Rank
leaf-score counts were 1,177,280 and 1,178,248 with zero overlap and a merged
count of 2,355,528. Receipt:
`results/flexhook/official_refer_kitti_v2.json`.

2026-10-22 — Implemented a separate `flex_tta` score-plugin line. The
FROZEN adapter is an exact score-only replay baseline. `ONLINE_STABILITY`
uses only the previous frame of the same tracker ID to update a bounded affine
score calibration; it never reads target GT or future frames and never changes
boxes or IDs. A first full run exposed a serialization bug where the official
car/pedestrian ID offset was applied only to the in-memory key. The duplicate
ID failure was recorded in the ignored `flexhook_score_tta_v2` run, then fixed
by applying the same offset to MOT column 2 and rerun as v3.

2026-10-22 — Full source-free score-TTA result accepted after the corrected
FROZEN reproduction matched the official metric. Across 862 expressions and
2,355,528 candidates, FROZEN scored HOTA 42.526 / DetA 30.629 / AssA 59.186;
ONLINE_STABILITY scored HOTA 42.525 / DetA 30.625 / AssA 59.191 after
2,153,705 causal pseudo-updates. The result is HOTA-neutral within 0.001 and
is retained as a baseline, not claimed as an improvement. Compact receipt:
`results/flexhook/flexhook_score_tta_v3.json`.

2026-10-22 — Remaining scope. The audited official FlexHook resources contain
only the Refer-KITTI-V2 checkpoint and Temp-NeuralSORT tracker bank. No legal
official FlexHook BDD/AerialMind weights or corresponding tracker banks were
found, and no cross-tracker full result has been claimed. BDD/AerialMind and
cross-tracker generalization remain unfinished; this log records the limit
instead of substituting an unverified model or target labels.
2026-10-22 — Completed the requested V2→Dance cross-domain test with the
second-stage weight fixed to `refer-kitti-v2-best.pth`. Because the official
Dance tracker bank was not available, an isolated image-only YOLO11x+ByteTrack
tracker was generated on all 25 DanceTrack validation videos (25,508 frames,
224,807 track rows); its receipt records `ground_truth_read=false` and
`expressions_read=false`. The first full inference failed on the released
loader's removed Transformers `batch_encode_plus` API. A project-local
`compat/sitecustomize.py` shim restored the legacy entry point and fixed
fixed-length truncation for `text_len=25`; a full tokenizer regression passed.
The second attempt exposed the same API's non-truncating behavior on long
subword sequences; this was corrected without changing model weights or data.

The complete two-GPU inference on physical GPUs 3 and 5 then loaded the V2
checkpoint successfully and processed 4,209 validation batches. It generated
25 videos × 17 expressions = 425 prediction files. The repaired post-hoc
TrackEval run completed on all 425 sequences and returned HOTA 5.2445, DetA
4.4082, AssA 6.2627, DetRe 59.591, DetPr 4.5369, AssRe 10.966, AssPr 12.786,
LocA 89.511, 2,791,694 detections and 5,607 predicted IDs. The low score is
accepted as the actual zero-shot baseline: V2's score gate is badly
miscalibrated on Dance candidates and association is weak. The released
`refer-dance-best` + DETR/NeuralSORT 32.165 HOTA number is reference-only,
because both checkpoint and tracker differ. The compact receipt and analysis
are `results/flexhook/flexhook_v2_to_dance_yolo11x_bytetrack.json` and
`docs/reports/flexhook_v2_to_dance_yolo11x_bytetrack.md`; the raw 426 MB
result JSON and TrackEval outputs remain on the local data volume. For
leakage accounting, the released loader opens `labels.json` unconditionally,
but its Dance test branch does not consume the labels; candidate boxes and
scores are derived from tracker files and images, and GT templates are used
only by post-hoc TrackEval.

2026-10-22 — Corrected the accepted Dance protocol to use FlexHook's
official Dance-specific candidate source. FlexHook code is open source at
`buptLwz/FlexHook`, and its official Baidu asset share provides the
`DETR-NeuralSORT-Dance` tracker bank. I downloaded and verified all 25
DanceTrack validation files: 211,085 MOT rows across 25,508 frames; the
download receipt records `ground_truth_read=false` and
`expressions_read=false`. This supersedes the earlier YOLO11x+ByteTrack
diagnostic as the proper official-tracker baseline; no YOLO output is used in
the accepted run.

The fixed V2 checkpoint `refer-kitti-v2-best.pth` was run on physical GPUs 7
and 5 with two processes, across all 25 videos × 17 expressions = 425
sequences. Post-hoc TrackEval completed 425/425 sequences. The accepted
result is HOTA 12.254, DetA 4.7043, AssA 31.95, DetRe 57.811, DetPr 4.8663,
AssRe 36.485, AssPr 65.379, LocA 91.376, IDF1 5.6069, with 2,524,962
detections and 10,652 predicted IDs. Relative to the earlier generic run,
official Dance candidates improve HOTA by 7.0095 and AssA by 25.6873 points.
The remaining gap to the official same-domain `refer-dance-best` reference
of HOTA 32.165 is retained as the expected V2→Dance domain/checkpoint
mismatch, not hidden as a TTA gain.

The official tracker downloader, compact receipt, and analysis report are
`scripts/download_flexhook_official_dance_tracker.py`,
`results/flexhook/flexhook_v2_to_dance_detr_neuralsort.json`, and
`docs/reports/flexhook_v2_to_dance_detr_neuralsort.md`. A relative-output
path bug in the TrackEval wrapper was also fixed and regression-tested with a
successful full rerun.

2026-10-23 — Completed the requested ceiling comparison with the original
FlexHook Dance second-stage checkpoint. The hypothesis was that
`refer-dance-best.pth` on the official Dance DETR+NeuralSORT candidate bank
would measure the same-domain upper bound, while `refer-kitti-v2-best.pth`
would provide the fixed-checkpoint V2→Dance comparison. Both runs used the
same 25 validation videos, 17 expressions per video, 425 sequences, no
target GT for inference, and post-hoc GT only for TrackEval.

The first reruns exposed a Transformers 5.4 compatibility failure rather than
a model failure: the local RoBERTa fast tokenizer loaded an empty merge table
and encoded words such as `person` character by character. The initial HOTA
8.224 same-domain and HOTA 12.254 V2 records are therefore superseded. The
smallest justified fix was a direct ByteLevel BPE construction from the
released `vocab.json` and `merges.txt` in `compat/sitecustomize.py`, followed
by a single-token regression and complete two-process reruns.

The corrected same-domain Dance checkpoint reached HOTA 32.134 / DetA 28.341
/ AssA 36.491 / IDF1 27.832. The corrected V2→Dance run reached HOTA 8.206 /
DetA 3.1857 / AssA 21.138 / IDF1 4.436. The ceiling-minus-V2 gaps are HOTA
23.928, DetA 25.1553, AssA 15.353, and IDF1 23.396 points. The same-domain
number closely reproduces the vendor's 32.165 reference. This comparison is
retained as an upper-bound diagnostic, not as a TTA gain claim. Compact
receipt and report: `results/flexhook/flexhook_dance_ceiling_vs_v2_corrected.json`
and `docs/reports/flexhook_dance_ceiling_vs_v2_corrected.md`.

2026-10-24 — Re-established the isolated FlexHook goal from the latest user
task.  The active project is `/data2/user/FlexHook-TTA`, branch
`research/flexhook-generalizable-plugin-tta`, and remote
`LYQ1107/TTA-RMOT`; no LocateMOT or GroundJEV work is being mixed into this
line.  The corrected tokenizer regression and GT-free loader regression both
pass.  The corrected target protocol remains V2 checkpoint → fixed official
Dance DETR+NeuralSORT candidates → post-hoc TrackEval.

2026-10-24 — Completed D1–D3 offline diagnostics before accepting TTA
results.  The fixed candidate bank has 95.749% IoU-0.5 and 85.071% IoU-0.75
recall, while the frozen V2 referring gate retains only 5.046% IoU-0.5
recall.  Score AUROC is 0.5397, average precision 0.6403, and ECE 0.5266;
therefore the failure is both calibration and cross-domain ranking, not a
simple score shift.  The 2.408% tracker switch rate is recorded as a smaller
temporal factor.  GT-based threshold results are marked development-only and
were not fed back into any run.

2026-10-24 — Completed full source-free score calibration controls on all 425
sequences.  FROZEN is HOTA 8.206, ONLINE_STABILITY 8.1741,
SOURCE_PRIOR_CALIBRATION 8.206, and QUERY_TRACKLET_CALIBRATION 8.2092.  The
best +0.0032 HOTA change is not a meaningful improvement and IDF1 decreases;
the compact record is `results/flexhook_v2_dance/calibration/calibration_v1.json`.

2026-10-24 — Implemented real parameter-level PCD adapters.  The residual
adapter is zero-gated at reset; LoRA replaces only Q/V projections in the
last two PCD blocks with zero-output rank-4 factors.  A rank-4 LoRA pilot
verified source-output MAE 9.70e-5, zero sign mismatches, and 13/13
nonzero-gradient updates.  The optional `TrackletLanguageEvidence` module
adds an independent RGB/HSV clothing-plus-causal-tracklet evidence path with
SUPPORTED_POSITIVE/SUPPORTED_NEGATIVE/UNKNOWN gating; it does not read source
scores or target labels.

2026-10-24 — Completed the full PCD Residual source-free run on 25 videos and
425 expressions using independent whole-episode shards.  TrackEval completed
425/425 sequences.  The result is HOTA 8.014 / DetA 2.4762 / AssA 25.938 /
IDF1 3.6957, versus frozen 8.206 / 3.1857 / 21.138 / 4.436.  Association
and IDSW improve, but detection recall, IDF1, and HOTA fall; the method is
retained as a reproducible negative result.  There were 108,511 optimizer
steps and 108,511 nonzero-gradient steps, with 80,880 selected rows.  Compact
receipt: `results/flexhook_v2_dance/pcd/pcd_residual_full_v1.json`; raw
predictions and TrackEval outputs remain under `/data2`.

2026-10-24 — The initial single-GPU LoRA full launch was stopped before any
formal metric after producing only 19 episode files; its partial output is
preserved as a resource/provenance artifact and is not included in results.
The formal LoRA run was relaunched with the unchanged rank-4, update-interval
4 protocol as two complete-episode shards on GPUs 4 and 7.  GPU7's unrelated
process was left running; no other process was stopped and no tracker or
dataset input was changed.

2026-10-25 — The formal PCD LoRA run completed normally (exit code 0) on both
complete shards: 221 + 204 = 425 expressions over all 25 Dance videos. The
merged GT-free audit passed with 425 prediction files and 82,248 checked rows;
target GT/labels and future frames were not opened, and official tracker
geometry and IDs were unchanged. TrackEval returned HOTA 7.8654 / DetA
2.3305 / AssA 26.547 / IDF1 3.492 / IDSW 138 / Dets 82,248 / IDs 1,313.
Compared with Frozen HOTA 8.206 / DetA 3.1857 / AssA 21.138 / IDF1 4.436 /
IDSW 259, LoRA improves association and switches but is a negative overall
result because detection and IDF1 fall. Compact receipt:
`results/flexhook_v2_dance/pcd/pcd_lora_full_v2.json`.

2026-10-25 — The first independent-evidence pilot command failed before
inference because a relative `--vendor vendor/FlexHook` path was duplicated by
the compatibility loader; no data or GT was opened. The smallest fix was to
use the absolute vendor path, after which a one-expression
`TRACKLET_LANGUAGE` pilot completed on GPU2. It processed 1,202 frames with
301 nonzero-gradient updates, emitted 1,090 independent negative evidence
labels and no positive/unknown labels, and kept target GT closed. The pilot
passed the corrected one-video/one-sequence fixed-candidate audit. The first
audit invocation also used its full-suite default (25/425) and failed only on
the expected coverage assertion; rerunning with explicit 1/1 expectations
passed. The pilot is evidence-method validation, not a full-suite metric:
`results/flexhook_v2_dance/evidence/tracklet_language_pilot_v2.json`.

2026-10-25 — Integrated Positive-Free preparation commit
896d67fe92d44e18c33980a7b26c2d3da1f8b7f7 in the isolated TTA-RMOT worktree.
The original research branch and all checkpoints were preserved. Initial
server tests were 21 passed / 1 failed because the one-video smoke fixture
implicitly invoked the formal 25-video loader; explicit fixture video
selection fixed this without weakening formal coverage. Caller paths are now
resolved before vendor cwd changes, Positive-Free checkpoint loading is
strict, and episode metadata is compatible with the existing shard merger.
Final specified tests: 26 passed / 0 failed; compileall, bash syntax and a
synthetic-only shard merge passed.

The expanded asset preflight verified the pinned V2 SHA256, all 25 official
Dance tracker hashes against their download receipt, all tracker-referenced
images, 425 expressions, Frozen score coverage, the full 17-video source
manifest (5,172 referenced images), source expressions/labels and the repaired
tokenizer. Source assets include 0017, but the vendor source-training list has
only 16 videos and omits 0017; no source training or vendor split change was
performed. The server-preflight development assertion initially used an
incorrect token ID; independent tokenizer regression caught it, and the final
check compares against the released vocabulary (person ID 5970).

Single-GPU sequential smoke on the previously idle physical GPU5 passed for
ZERO_UPDATE, PF_RANK, PF_IDENTITY, PF_HYBRID, PF_RANK_GUARD and CPU score replay
RELATIVE_ONLY. Each invocation used exactly one expression and 12 frames.
Each learning method performed 3 finite nonzero-gradient updates. A separate
norm-bound probe rolled back all 3 attempted updates; a no-evidence probe
performed 0 gradient attempts and 0 optimizer steps. Episode reset and causal
history checks passed, frozen model parameters had no gradients, and dynamic
file auditing observed 0 Dance GT open attempts. ZERO_UPDATE had 0 source
selection/sign mismatches (maximum margin error 1.64032e-4). All methods kept
this all-negative-source clip empty, so no forced positive was introduced;
nonempty real prediction row coverage was not established by this clip.

The four-GPU launcher was exercised only in dry-run/locked synthetic tests.
Its whole-video shards cover 7/6/6/6 videos and 119/102/102/102 episodes.
At completion only GPU5 was truly idle; other jobs were preserved. No full
source training, full target inference or TrackEval was started. Code-ready
does not imply measured accuracy gain or availability of four idle GPUs.
Receipt: docs/FLEXHOOK_POSITIVEFREE_SERVER_SMOKE_20261025.md and
results/positive_free/server_smoke_20261025/validation_summary.json.
# 2026-10-25 Positive-Free full-suite execution start

- Engineering follow-up: 33 focused tests passed; cache-enabled four-method smoke passed. LPT workload equals the requested 52,688/53,495/52,621/52,281 row groups; dynamic launcher rejects occupied GPUs and writes restartable episode receipts.
- Scientific input defect found before full launch: old Frozen has 210,393/211,085 observations. Vendor exclusive range omits 692 final track observations (including 14 single-observation tracks). Four source-only GPU probes returned clearly positive margins on omitted rows; this is not numerical cache noise.
- User explicitly chose complete candidates + rebuilt matched Frozen, not strict historical 8.206 sampling. Original Frozen/hash preserved. Separate same-V2 supplement has 11,764 true logits, 384 positives, SHA256 `3693edc6ef153b8ce0ca87aeec1b196d6b360ff775b8fe44755c4b15d7397639`. Preflight now requires exact score/tracker observation coverage and passes with this supplement.
- Full runner uses per-episode seed 0 reset for schedule-independent recovery, no changed algorithmic hyperparameters. Frozen LRU limit is 16 GiB/worker to retain the longest video across expressions; GPU admission requires >=24 GiB free and no compute processes. All target labels stayed closed.

- User explicitly authorized full V2→Dance online TTA and sealed-prediction official evaluation, but not source retraining or sharing occupied GPUs.
- Project is TTA-RMOT at `/data2/user/FlexHook-TTA`, independent of LocateMOT/GroundJEV. Verified remote prep HEAD `a0cb27e`; created the requested separate full-run branch.
- Initial resource snapshot: only physical GPU5 has no compute process; other GPUs belong to existing jobs and are not interrupted.
- Hypothesis: cache only frozen multi-scale backbone/text outputs; retain original language-conditioned C-Hook/PCD, causal history and FP32 decisions. Validate real logits, margin signs, selected IDs and LoRA gradients before formal use.
- Plan: weighted whole-video shards, atomic episode receipts/resume, full ZERO_UPDATE baseline gate, then all five controls/adaptations, official evaluation and offline attribution. No pilot-only completion claim.

## 2026-10-25 TTA-RMOT six-method full research closure

- The same full-suite shell finished exit0 after PF_HYBRID. Six methods each cover25 videos/425 expressions/3,588,445 candidate-query rows/433,636 active frames. Completed predictions were not rerun; no source training or foreign process kill.
- Matched ZERO baseline8.2180; Relative7.5705; Rank8.2433; Guard8.2444; Identity8.8179; Hybrid8.3213. Old8.2060 is historical only. Fixed V2/MOT/default rank4/lr5e-5/interval4/seed0/FP32 unchanged.
- Hybrid recovers986/loss1 GT observations, FP net+3,832 (newFP3,861/removal35 are distinct), EmptyFP90,852, AUROC0.647115.107,869 accepted updates/200 restored norm rollbacks;425 pre-episode resets and25 strict loads pass, no frozen gradients. Actual7.6323 GPU-hours/7,193.22s wall.
- Identity remains the strongest point estimate(+0.5999), but its video95% interval[-0.083812,+2.430322] includes0; ranking and Empty rejection worsen. Hybrid does not complement the stronger single module(-0.496598 HOTA versus Identity); no success claim based only on extra losses.
- Completed all-six Query-Present/Empty, candidate Recall@K versus any-hit, video-cluster bootstrap and Hybrid pair comparisons. Static figure rendered/visually checked; final report includes recovery-versus-FP, compute dependencies, A+B primary failure decision and caveats. Guard cost includes Rank and112.43s GPU equivalence, not just7.34s CPU derivation.
-52 focused tests pass(10.85s); compileall/shell syntax pass. Compact recovery index records all2,550 sealed receipts and actual model checks; missing full-run post-reset smoke flags remain null, not invented passes.
- Keep all six full results. Next authorized phase is bounded query-presence evidence design/prototype only: no seventh full target run, source retraining, GT tuning, forced positive or external semantic teacher. If no independent semantic signal is found, report UNKNOWN/skip rather than stack losses.

## 2026-10-25 Query-presence prototype preparation (after six full studies)

- Design: frozen four-branch interaction J=S(V,Q)-S(V,Q0)-S(V0,Q)+S(V0,Q0); same original decode, same geometry/IDs, Q0=person and zero FEATURE reference. Algebra removes additive visual/query offsets, unlike source-rank preservation or physical-ID InfoNCE.
- Feasibility boundary: positive/stable interaction can still be semantically wrong; zero features are not a labelled empty image, generic noun is not a negative query. No independent validated semantic-presence supervision has been found. Keep UNKNOWN, no prediction override, no hard pseudo label, no new loss/optimizer. Preserve empty/multi-target frozen fallback.
- Implemented causal bounded history and <=1query/12frame source-only frozen GPU probe. Six targeted scientific tests pass(0.03s), compileall passes. Tests include stable interaction still UNKNOWN and zero interaction not proving absence.
- GPU smoke remains to execute on one memory-fit/few-process card after all full-suite workers ended. Do not confuse prototype code readiness with measured semantic gain or launch a seventh full experiment.

## 2026-10-25 Query-presence prototype smoke and final decision

- Real GPU0 selected by memory fit/fewest processes (0 processes/81,151MiB free; GPU5 tied), after all six full runs ended. Same strict V2,1expression/12frames/36candidate observations; keep original `dacning` query spelling.
- PASS_FROZEN_DIAGNOSTIC_ONLY: source branch margin maxerror6.96182e-5/0 sign differences; person token5970; FP32 finite; parameter versions unchanged/0trainable/0updates;0GT open attempts; causal history/reset pass.5.8613s wall/1.5791GiB peak Torch allocated. Raw receipt/log kept, no seventh formal TrackEval.
- Interaction ranges[-12.658524,2.380775],5/36 positive; one provisional supported ID at frames11/12, but all statuses remainUNKNOWN and no prediction override occurs. Offset cancellation error<=1.43051e-6. This proves nontrivial diagnostic execution without forced positives, not semantic correctness.
- Independent reliable semantic presence supervision was not found. Keep the audit prototype, do not promote it to a validated learner, do not stack another loss or start a full run. Six full studies and bounded prototype are now finished; wait for explicit further authorization.
- All58 targeted tests pass(11.09s); compileall and both shell syntax checks pass. All actual full-run source trees are verified public; prototype e6d2d744 corresponds to identical published tree at47e3983716c2a4fd28936d2981c9647a4d3ba7a9.

- Final delivery preserves complete text logs, not just excerpts:138 preparation/scheduler/worker/model/official TrackEval/prototype logs,10,482,502 raw bytes,673,730-byte gzip tar. Archive contents were re-read and verified file-by-file; images/weights/prediction-margin streams remain in the recoverable server roots. Twelve actual full/prototype code commits map to identical published trees; vendor patch unchanged. No additional full inference/training launched.

## 2026-10-25 Fast recoverability diagnosis (new user-authorized goal)

- Project/branch/remote confirmed TTA-RMOT only, public29f265c/local41b5819 trees already verified; six full methods remain immutable. No source training or repeated full inference.
- Plan0: compare25/425 expression/GT/seqmap files against original Refer-Dance archive, raw labels membership/boxes, one-based images/tracker frames and official mark/class rules. Report any discrepancy before altering a protocol.
- Plan1: CPU-only POST-HOC Count-Matched Frozen, take Frozen top-k for each frame with k from sealed Identity output (not GT). Evaluate all425 in a separate process; attribute reproducible HOTA gain to count conditioning without claiming a causal percentage or deployable method.
- Plan2: full17-query source score matrices, fixed episode-presence statistics, all-other and GT-disjoint text swaps, row-mean visual bias/random controls, per-video/expression results and video-cluster uncertainty. GT never selects an online threshold.
- Conditional Plan3: only if score diagnostics cannot establish frozen internal clothing information, use fixed GT-independent sampled frame quantiles across25 videos/all17 queries; frozen source only, first real-frame functionality check. Retrieval/probes are offline diagnostic; grouped video CV and query/background controls prevent trivial semantic claims. No probe weights enter TTA.

- Protocol-auditor implementation correction: the first diagnostic incorrectly interpreted labels.json normalized xywh as center xywh, producing40 spurious bbox discrepancies. Official labels use normalized TOP-LEFT xywh. Archive CRC, frame-ID/GT-ID membership, seqmaps and40/385 counts already agreed. Correct only the new auditor parser; preserve its first diagnostic/log. Original labels/GT/predictions/evaluation are untouched.

- Phase0PASS:852 archive metadata/CRC checks, all212,541 GT frame/ID and normalized-box matches, both425-entry seqmaps, historical GT links, one-based frames; allGTmark1/class1.40/385 is genuine original protocol, not evaluation filtering error.
- Phase1 completeCPU425: Count-Matched Frozen8.8206/Identity8.8179/Frozen8.2180. Same114,648 outputs/framewisek; gain+0.6026 versusIdentity+0.5999, descriptive ratio100.45%, not causal or deployable fixed threshold. Identity-minus-countmatched=-0.002684 unrounded,25videoCI[-0.027423,+0.012497]. Counts+Frozen order reproduce essentially all gain. No GPU call or GT access in prediction construction.
- Phase2 completeCPU3,588,445 scores in34.94s: presence maxAUROC0.725779/AP0.199373 vsrandomAP0.094118; mixed-frame sourceAUROC0.559705 vsvisual0.504819/other-text0.500284. Correct-queryTop1 minusother0.041003(CI[0.014186,0.065544]). Query marginal accounts65.31% within-video score variance, not proof that all marginal signal is wrong.
- Phase3 necessity: outputs do not establish whether latent clothing information is absent or merely poorly read out. Freeze fixed75-frame sample(10/50/90%MOT indices,all25/all17) before feature GT evaluation, with generic-person PCD and background/geometry/query controls. Only this small extraction is authorized; first functionality smoke, no LoRA or teacher. First9 focused tests include known bbox-format failure, count controls, overlapping text swaps and video exclusion.

- CPU phase verification:9 targeted tests pass(0.53s). Count-Matched andIdentity share114,350/114,648 selected observations(99.7401%), only296 frames differ,298 exchanges per direction, counts equal in every frame. Fixed sampling manifest generated from tracker frames and sealed source sentence/margins only; first video frames121/602/1083. No persisted full PCD feature arrays were found in sealed Zero/Identity roots.

- Frozen Phase3 extraction complete: initial real1frame/all17 smoke5.93s; full75frames/25videos/643MOT observations57.63s on GPU4,1.89GiB peak Torch, reused first smoke frame.0 source-margin error/sign mismatches,0 GT opens,0 trainable parameters/updates, all parameter versions unchanged. Actual sourcec6250299 has identical published treec8b679ed.
- No-training cross-video clothing NN:603 GT-matched annotated observations, generic-personPCD50.91% vsrandom32.75%, deltaCI[8.40,29.16]points. Background38.54%; PCD-minus-background delta12.37points has CI[-3.57,30.23], so cannot exclude scene confounding. Only5/17 query classes span>=2sampled videos; no claim of exhaustive fine-grained semantics.
- Additional PROBE necessity: above scene uncertainty leaves foreground clothing/readout recoverability unresolved. Execute only pre-specified CPU linearC1/seed0/5-fold video CV, no hyperparameter sweep or weight export. Compare query-only/background+text/geometry+text, and mixed-candidate WITHIN-frame AUROC where query and whole-frame context are fixed. This is supervised diagnosis, not Source-Free TTA.

- Probe environment correction: first invocation failed before fitting because dfot has no sklearn. Preserve its traceback; install pinned optional CPU dependencies in a separate data2 diagnostic-pydeps directory, without replacing numpy/scipy or the original inference environment. Code records fit folds, video separation, package versions, actual source commit and wall time. No failed sample is skipped.

- Fixed5fold/video GT diagnostic completedCPU133.06s,20fits,C1/seed0/no sweep/no saved weights. PCD pooledOOF AUC0.802146/AP0.241785, but query-only/background controls also have high pooled metrics. Within same69 mixed frame-query groups(11videos), PCD0.586402 vsbackground+text~0.5, deltaCI[0.028432,0.166747]; frozen original head0.616484 is BETTER than probe. PCD-minus-source=-0.030081,CI[-0.089035,+0.026710]. Do not call pooled0.802 a TTA gain or semantic recovery.
- Constant text/background readouts have within-group spread<=1.78e-15; raw strict-rank AUC~0.5008/0.4992 reflects dot-product roundoff, not candidate information. Preserve all10931 OOF rows and69 per-group comparisons. Probe conditional-video bootstrap is descriptive, not refitted independent confirmation.
- Final decision: output-count/calibration explains essentially all observedIdentity HOTA gain; official Query-Empty error and weak query-conditioned ranking remain. Frozen appearance/PCD is not wholly informationless, but decodability does not establish source-free identifiability; no independently reliable semantic pseudo-label signal found. Keep UNKNOWN/causal consistency as diagnostics only, do not start fullTTA or add losses.
- Delivery: all25/425 CPU analyses, fixed25video/75frame/all17 frozen features, grouped diagnostic probe, standalone report and visually checked static figure. Archive all425 Count-Matched prediction files WITHOUT GT/margins and20 complete diagnostic/model/failed-attempt logs, verify every member.67 targeted tests pass11.62s; compileall/bash syntax pass; original six published result files unchanged versus29f265c. Raw feature cache remains recoverable ondata2, excluded from GitHub. Stop after publication.

## 2026-10-26 New authorized Offline UDA goal

- Explicit protocol change: supervised fullV2 + unlabeled DanceTrain, not source-freeTTA; keep old six/diagnostic results. New branchresearch/flexhook-v2-dance-uda-20261026 frompublic5db7dadd.
- ActualV2 SHA rechecked71dbd691...613. GPU snapshot5free,3/4/6 oneprocess/memoryfit; no task killed, no resource reserved yet. Formal four-method runs require fresh admission and complete train assets.
- Prepared all40Train images from existing official11GB archive,41796frames, WITHOUT reading/extracting target GT payloads. Input-expression inventory is1560Train episodes, not680/17-per-video.25Val425 preserved. Three separate manifests strip target membership/presence and separate evaluator GT paths.
- Key asset missing: all40Train DETR+NeuralSORT predictions. Existing official bank has25Val files; original Refer-Dance ZIP has no tracker/model payloads. Search public official bank and exact detector/NKF weights/config; cannot export GT or silently train onVal/alternatefront-end.
- Source split discrepancy to resolve explicitly: TempRMOT officialfull train manifest17videos includes0017, while vendored FlexHook VIDEOS kitti-2 default lists16and omits0017. Complete-source task will use authoritative complete17with source annotations, preserve four source-val videos, and disclose the difference rather than silently discard0017.

## 2026-10-26 25Val transductive UDA diagnostic completed; formal Train→Val remains deferred

- Scope was explicitly changed by the user to first inspect25Val, not generate40Train MOT. Preserve original formal objective as incomplete; no target GT enters training/model selection. This is same-set transductive diagnosis, not source-free online TTA or unseen-video UDA.
- All4 methods each completed8973 finite/nonzero LoRA updates, source17video/all8896queries plus77track anchors, with0skipped samples and3same-sample AMP backoffs each. Sparse coverage is393unique source anchors and75target anchors/1275query windows, not all-frame optimization. All4full425predictions, per-video parity and official TrackEval completed.
- HOTA: Frozen8.2180, Source7.4352, Teacher8.2494, Weak6.9281, Combined7.7293. Teacher Count-Matched8.2479 reproduces almost all tiny point gain; Teacher video-clusterCI[-.205174,+.416379], adjusted residualCI[-.010013,+.019832]. No causal percentage or scientific success claimed. Combined fails to complement the better module.
- Frozen candidateAUROC.650204 falls in all4 methods; all candidateRecall@1/3/5 and correct-queryTop1 point estimates decline. Teacher mixed-frame query-minus-visual AUROC delta-.009390,25-video descriptiveCI[-.018439,-.001956]. Main bottleneck evidence remains referring/presence/calibration, not simply physical identity.
- Offline-only pseudo quality after all predictions sealed: Teacher POS84/958(8.768%), Combined84/878(9.567%);862/786POS exposures in Query-Empty. Repeated exposures are correlated, not unique people. No independent valid semantic teacher signal; UNKNOWN never relabelled negative or GT-corrected. C/D effective target presenceP/N0 and temporal0; legal relation evidence is active but does not establish positive existence.
- Final4GPU-process training+inference cost about5.7178h, actual overlap pipeline7319.48s; sharedGPU costs and failed-attempt lower bounds disclosed. Raw checkpoints/scores staydata2, SHA/recovery paths public. Four850-member archives contain exactly425predictions+425receipts each, no GT/images/weights. All code/results/logs published; README/progress/final generator corrected to show final state rather than old2-method interim.
- Retain code, negative results, original6studies, v1revoked smoke and all failed/resumed attempts. Stop this authorized stage; no fifth method, new teacher, Val GT parameter tuning or40Train run without further user direction. Engineering completion is not validated scientific improvement and not completion of the original formal40Train→25Val goal.
- Delivery recheck:52 focused PF+UDA tests passed10.89s(one non-failing anyio pytest-rewrite warning); compileall andUDA shell syntax passed. Regenerated final report from immutable metrics plus completed text/pseudo diagnostics. Complete raw text-log archive includes original model log_rank files as well as worker/launcher/recovery/TrackEval outputs; no metric, seed, checkpoint or prediction changed during delivery.

## 2026-10-26 New authorized P1 frozen semantic teacher investigation

- New explicit task replaces the prior execution direction, not its scientific history: audit YesTrack/Qwen3-VL-2B baseT0 and officialKITTI-V2 LoRAT1, then actual matched-budget25Val teacher inference, no student training. The app refused a replacement goal because the old formal40Train goal is unfinished; do not falsely mark that goal achieved. Track this new task in its requested independent branch and records.
- git fetch --all --prune succeeded but the configured fetch refspec omitted research branches; explicitly fetched the requested research ref. Actual remote5c72733290c455eaeaf0efe942f56ec6de8d9549 and tree29bbd48c5b20412ad18c828e0740469635252c46 verified, then createdresearch/v2dance-evidence-teacher-p1-20261026 from that public commit.
- YesTrack actual sourceed42b8315b28c94dc595d83b0d49bf57002bb281(2026-07-22),MIT. Its default from-MOT script points toV1 LoRA; must override with auditedv2bestweight. Official release is publicBaidu; correctV2 training negatives were downsampled, so score calibration differs. NoDance LoRA orGT used.
- BaseQwen/Qwen3-VL-2B-Instruct pinnedHF revision89644892e4d85e24eaac8bacfd4f463576704203,Apache2.0,public/non-gated; downloading todata2. FreezeT0/T1 numeric config and original320px ROI/20%margin; actual weights/hashes still to verify before GPU readiness.
- Registered GT-blind fixed1/5-anchor sampling, A/B/C/D plus budget-matchedB-plus, originalTCP/TRP andSoftMatch screening controls; explicit source-dev reliability grids and target quality/coverage/cost Gates before new targetGT metrics. No independent semantic benefit assumed from decomposition or image sensitivity.
- ActualT0 downloaded4,255,140,312bytes andSHA7de1838c...77a0matchesofficialLFS. OfficialV2 publicBaidu archive listed178MB but normalanonymousbinary download lacks signature; local search found only unrelated self-trained adapters. User explicitly allowed missingT1 if unavailable; no substitute used. User also approved13frame causal-window originalTCP/TRP with disclosed reset.
- Fixed sampling completed: smoke1649/1700cap, full8381/8500cap, both25videos/425queries, zero silently omitted empty cases. Source calibration64windows yields203legal source candidate-query observations; source truth remains outside teacher staging andGit.
- RealGPU0 functionalityT0/YES_FULL2pairsPASS:16.70s worker,1.7084s actualforward,4.020GiB peak allocated, originalYes/No function, cache persistence, kernelLandlockABI1,0targetGT opens, initial/finalfullparameterSHA9a13380b...a956equal,0studentupdates. This isnot yet fullsmoke orP1 quality. Code/sourceafdce21 hasidentical publictree at8cce99dc194fcd6e289905b85fe3c65403ab7de5.
- Firstrealrun exposed onlykernel-cache path warning underfilesystem restrictions; route runtimeCUDA/Torch caches to own writable output, do not broaden annotation access. Pre-batch independent frozen calls across candidates withsame inputs, batch8within16GiB budget aftermeasured4.02GiB firstsingleview; no thresholds/prompts orresolution tunedfromGT.
- Allsix4-pair functionality workers passed onphysicalGPU0/5/7/2 in55.31s parentwall, includingB-plus and13frameTCP/TRP; noGTreads, parameterupdates orotherprocessstops. Full targetprobe hasnot yetrun.
- Source calibration firstattempt stopped onmissing stagedsource0005/000072.png: previous-observation four-frame control needsoneearlierimage than currentfourframehistory. Fixedpreparer andadded38sourceimages, preservingall203sampleIDs/manifestSHA b888b33f...8fc8 andsource-truthSHA89d61067...c48. Addedtargetedhistory regression;6tests pass2.80s. Preservefailedstdout/cache andresumeonlyfailedsourceunit; no easier sampling orGT-derived target selection.
- Source calibration resumed successfully299.65s onGPU7; 203pairs/82sourcepositives. Source-onlyfrozen thresholds0.95/0.05; originalpositivegrid12accepted/10correct at0.95, verifier5accepted/5correct but5/82coverage. Smallgeneric-sourcecontext doesnot validateDanceclothing. All25/425fullsmoke started onGPU7/5/2/3; no newtargetGTread orstudenttraining.
- BeforetargetGT evaluation, implemented independentpredictionseal/hash/index andcandidate/one-to-oneGT/presence/queryswap/pervideo/clusterbootstrap analyses, withconditionalgates andreportgenerator. SoftMatchcontrol usesonlysource-frozenhalfGaussian,n_sigma2,no newhardthreshold; weightedcontrol doesnotcreate semanticinformation. Addedfocusedmetrics/ties/duplicateMOT regression;9tests passed3.23s. Noqualityresultclaimed beforecompletepredictionseal.
- P1fourGPU throughput preparation (not numericalmethodchange): completeprobe partitions all25wholevideos GT-blind into2040/2040/2295/2006fixedpairs; eachshard runsall6methods withoriginalsampleIDs andcompletecausal13framewindows. Avoididlecards aftercheapmethodsfinish. Runninghealthysmoke untouched. Addedexactmerge/receiptprovenance, preservedcompletedshards andseparateworker/wall costs.10focusedtests pass3.16s; dry-plan verifiedall25videos/8381pairs once, noGTopened.
- Complete25video/425querysmoke ALL6PASS, each1649pairs, parentwall2695.1017s. B-plusactual2116.8380s; TCP1118.7465s; rawcache hitsdoNOT makeindependentmethodfree. AllfullparameterSHAsunchanged, GTopenattempts0, updates0. Fullprobe admissionselectedphysicalGPU6/4/7/5, whole-video4shards nowrunning. Actualsourcefc1c308 hasidentical publictreed8ab11e535aaf3a4f1daf3618b27c9572e518577; no P1 targetGTmetrics yet.
- Exact4-way syntheticmerge andfull-samplecoverage tests now11PASS3.28s. Completedsourcephase resume returnsPRESERVED_COMPLETE_PHASE withoriginal299.6531swallandnoGPUworker; originalphasefileSHA unchanged. SecondaryDvsC countmatching andparser-typo subgroups registeredbeforetargetGTevaluation; no thresholds/prompts changed.

## 2026-10-27 P1 continuation, not a new experiment

- Existingfour-shardprobe continueswithoutinterruptinghealthyworkers. BeforeanynewP1targetGTscore, addofflinevisualfeature summaries andnon-Yes/Nocontrolcounts groupedbycomplete-queryofficialIoU, explicitlyNOTattributeGT. No teacher/rule/prompt/threshold change.
- Capturedactualteacher runtime packageversions andpinnedYesTracksourcefileSHAs inassets/runtime_versions.json; readonlybaseenvironmentunchanged. Historicalresults/positive_free,results/diagnostics,results/uda arebyte-identical ingit versusUDAstart5c727332. Source/frozen/oldTTAresultspreserved.
- BeforeGT evaluation, addcomplete-queryscore-bottleneck counts (ACTION/APPEARANCE minima, notattributeaccuracy). Separateadded/removedone-to-oneFProws fromexisting-rowFPrematching; a duplicate-boxregression checks a newlyaddedMATCHEDrow can turnanoldrow intoFP. No prediction/rulechange; sparseIoUdiagnostic only, notTrackEval.
- ActualcompletedD/shard03 exposedpreprocessingoverhead:2006pairs2931.58s worker,612.25s modelforward. AddedreadonlyRGB/mean/patch-SHA LRU cachingonly; 40real-frame/view/variantcold+warmchecks across2fixedsamples EXACTRGB/metadata/SHAparity,0GPUcontexts/GTopens. Healthyworkersnotautomaticallyrestarted; alloldtimings retained. Cacheversion recordedperworker, so walltimeisnot a controlledalgorithm-onlyspeedcomparison.
- AllfourDshardsfinishednormally. OriginalGPU7 nextB-plusfailedatpre-modelresourceadmission (16GiB+4GiB threshold), no prediction skipped. Independentrecovery ofonlyshard02 nowonphysicalGPU2,excludinghealthy4/5/6; measuredworstsmoke5.15GiB justifies8GiBallocation+4GiBmargin. SameBF16/batch8/320px/prompts/source-thresholds; max4GPUs,otherjobsnotstopped. RepaircontrollerPID1778432,worker1778442; durableattempt/logs underfull/T0. Optionalrestartrequestdoesnotpausehealthyqueue.
- Recoveryshard02 COMPLETE: all6methods validated2295pairs/7videos/119queries, allmodelSHA9a13380b...a956unchanged and0GTopens. A/B/C/D preserved; onlyB-plus2071.89s andTCP1159.00s computedonGPU2. The separateGPU5shardalsofinishedall6normally. No healthyworkerwasrestarted; nowonlyTCPshards00/01 remainonGPU6/4,withnewreadonlycacheloadednaturally.
- GlobalA/B/C/D/B-plus merged8381pairs/25videos/425queries each, hashes/frozenreceipts verified; no targetGTread. Thisis fivecompletedpredictionsets, not yetglobalP1qualitysuccess. Independentrawcache/predictionseal andofflineevaluation waitforlastTCPcompletion.

## 2026-10-27 P1 frozen teacher study completed; stop before P2

- All six T0 methods completed and merged, each8381pairs/25videos/425expressions; fullsmoke each1649. The original controller exit1 records the retained GPU7 admission failure, not missing final results: only failed B-plus and unstarted TCP/shard02 were repaired, all healthy work preserved. Full pipeline wall11982.33s; no other task stopped and max4GPU concurrently.
- Sealed all predictions/configs/raw index at2026-10-26T19:18:34.969686Z, then independent targetGT evaluation started19:20:01.905484Z. Teacher GT-open attempts0, future requests0, parameter updates0; full9a13380b...a956 unchanged. Exact B/C and D/B-plus visual/call mismatch0. Raw204327responses and SQLite retained ondata2; public outputs are compact GT-free predictions.
- Official40Present/385Empty confirmed. Primary193GT-present frame-queries contain761sampled pairs/485positive candidates/1102GT observations; one-to-one recoverableGT482. Different candidate prevalence from all8381 is disclosed, no independent clauseGT/spatial claims. All425 auxiliary,425query,25video,query-swap,visual controls,UNKNOWN,FP-row/rematching,precision-coverage and costs are recorded separately.
- Primary AP Frozen72.929%, A89.771%, B88.576%, C/D88.725%, Bplus88.536%, short-windowTCP84.393%. A-minusFrozen+16.841points, videoCI[+2.461,+31.137]; promising generic teacher signal, NOT new RMOTHOTA/student success. A source-gated primaryPOS precision90.370% and GTcoverage33.122%, versus full425 officialPOS precision57.911%.
- C-minusB AP+0.149points is not sufficient contribution. D and C rawrank identical; at D's182accepted primaryPOS, C top182 has the same168true candidates (precision difference0, videoCI[-2.315,+2.839]points). D coverage168/1102=15.245% versus A365/1102=33.122%. D/A matched-count precision delta-1.099points, CI crosses0. B/Bplus have insufficient accepted counts for the registered one-direction coverage comparison; report inconclusive rather than proof of inferiority.
- GateA supports simple T0 teacher, Evidence GateB/C fail; no independent benefit established for decomposition/verifier. Add only offline paired AP/Recall/precision contrasts using the exact existing25video bootstrap draws; no prompt, gate, threshold, model, candidate, seed or sealed prediction changes after GT. SoftMatch-inspired accepted weights are1 and duplicatesA, not an independent teacher. T1 officialV2 remains unavailable/user-waived; do not substitute unrelated adapters.
- Retain code, negative results,102-member full text-log archive and all recovery attempts; source203pairs used fixed0.95/0.05 only, no targetGT tuning. Allpipeline stage allocations12.6485GPU-hours (shared-worker time, not exclusive energy), first2pair functionality16.70s separately. Cacheversion/hardware differences prevent controlled speedup claims.
- STOP_AFTER_P1: no student/source/TTA long training started. Formal40Train predictedMOT and independently verified10video referringGT still missing; conditional readiness docs do not imply executableP2. Historical sixTTA/diagnostics/UDA tracked results unchanged versus5c727332.
- Final verification:12 targeted tests passed3.08s, compileall/bash syntax/diff checks passed,40 real RGB/metadata parity checks retained, all102 archived text members SHA-verified. All50286 compact method predictions,3400query summaries/400video-scope summaries, paired uncertainty, full visual diagnostics, costs, figures,16answers and recovery paths published at0685e51257494aab163720e47205fcdd3491bab1, exact localresulttree841bc83ee2807cc216353adfe43d13d47439f7e4. publication.json records source/public mapping; no new GPU job.

## 2026-10-28 New MLLM raw-soft distillation goal; user changes first stage to Val

- New explicit Goal created, independent branchresearch/v2dance-mllm-naive-distill-20261028 from verified public7026d82c; clean source tree9d0448a3 matched previous localP1. P1 and all older TTA/UDA reports/caches/checkpoints preserved.
-40Train MOT directory still missing; code/public source audit finds FlexHook inference consumes provided tracks, not a detector/NKF generator. Cloned officialiKUN read-only as upstream evidence, no substitute frontend or GT tracks. User explicitly says firstuseVal, so current stage is25Val same-set unlabeled distillation diagnosis, not Train→Val. Shared GPUs reauthorized, memoryfit/fewestprocesses, max4/no other job killed.
- P1 A rawsingle-frame output8381pairs/25videos/425queries/2125windows exported and sealed with full logits, originalBF16p, validity, images/ROI/prompt/bbox/model and identity-qualified keys. All8381 valid,980p>.5/7401<=.5; no GT labels, no hard screening. P1 thresholdUNKNOWN is not converted toNEG; all valid originalsoft evidence retained. NewteacherGPUcalls0 but inherit/disclose P1teacher cost.
- Implement independentMLLM_DISTILL, final-layerYes-No margin and per-personsoftBCE plus unchangedsourceCE; rank4last2Q/V24576params,lr5e-5,seed17,lambda1fixedand source-onlychecked.19 targeted tests passed3.96s. No source trainer rewrite or P1 file modification.
- First interface failures (CUDA init under Landlock, user-site runtime dependency, omitted existing tokenizer shim, provenance/source symlink paths) occurred before updates and are all retained. v6interface passes8windows/31candidates,0steps,nonzero finiteKDgradients,strictV2/frozen-equivalence/token5970/checkpoint roundtrip/temp cleanup/GT0. Legal shared KITTI images/expressions are resolved read-only data aliases, not another project's model/code/task.
- smoke_v1 had6realsteps but a source glob authorization omission silently omitted expressions; sourceval becameempty/NaN. Explicitly REVOKED, checkpoints never used as formalinitialization. Fix source-expression resolved whitelist and assert8896expressions/8973sourcewindows/64sourcevalwindows plusfiniteCE BEFOREsteps. No skipped data, altered coverage or NaN masking.
- smoke_v2 passes6actualupdates/6nonzeroKDgradientsteps,41sourcecandidate exposures/22validtargetexposures,deltaL2.027242,sourceplan8973,peak5.247GiB,GT0/frozenunchanged.12baselineauditchecks pass: E0completeofficial25MOT211085rows; historicalE1 has exact sourceplanSHA/seed/adapter/optimizer/steps/source-onlyselection and targetoptimization0, so reuse its real7.4352 and cost rather than retrain. MeanTeacher8.2494 is historicalreference withdifferenttargetsampling/supervision, not a matched-budgetMLLMcontrol.
- FormalE2 and full425evaluation are next, not already complete. Only actual fullRMOT/ranking/quantity-adjusted evidence can authorize subsequent conventional controls or innovation; do not call sparse AP a HOTA gain.
# 2026-10-29 PSR 主训练完成与匹配对照队列

6/8学生完整425预测已封存，Geometry修复父进程87918正常exit0。两组剩余学生各到6000步，仍原预算。
在等待其余GPU工作时，CPU已完成前六组12个完整双向数量对照；无新GPU调用/无重跑预测/无目标GT打开。
部分预计算只保存control_prefills收据，不创建最终controls_sealed；GT评测仍待全8组完成。

实际接力成功：Uniform425推理退出后，原GPU1启动NaiveTemporal PID3256288；Discovery425推理退出后，等待器29327在GPU6启动MeanTeacher PID3264581。
另两卡4/5仍为Geometry/Confidence推理，合计4物理卡；没有停止其他任务，也没有重复已完成episode。

已将剩余匹配Mean Teacher接力写成单个CPU等待父命令：仅在Discovery完整425预测封存且其GPU进程已由父进程wait返回后入场。
保留原1/5/4卡队列，最多新选1卡，重新检查显存/进程数；目录锁在CUDA分配前去重原队列稍后到达的同项，不迁移健康训练。

训练锚点GT-free分组检查（无新GPU、未读GT）：Naive教师POS656中学生选40（6.10%，源55），PSR657中学生选47（7.15%，源45）。
POS BCE虽从8.45/8.64降到1.99/1.97，逾92%教师POS仍被固定decoder拒绝；总体BCE下降不能直接证明正支持恢复。
仅作已登记可学习性证据，所有后续对照仍原配置，最终结论等待封存后完整RMOT/覆盖/排序/数量归因。

后续实测：Naive/PSR各425完整推理已封存，耗时1851.47/2021.72秒，均source-val选8973步，GT0。
Discovery和Geometry首次入场失败，根因不是模型：原GPU2/0被新外部进程占用，实际剩余18692/18690MiB，低于原20GiB入场预算。
保留16GiB allocator+4GiB headroom，不降低标准、不停止他人或健康Uniform/Confidence。
增加修复子队列显式排除健康GPU1/5并计入四卡总上限；重新审计后仅在其它显存足够、进程较少的两卡恢复这两项。

项目仍为TTA-RMOT；按用户明确授权使用25Val无标签同集适应，不是40Train→Val。
Naive/PSR主训练各完成8973步、17源视频完整覆盖；8500直接标签均实际访问，PSR254条独立弱插值均访问，目标GT打开数0。
训练主体耗时2554.08/2391.76秒；不把loss/梯度当成科研成功，尚无本轮新HOTA。
修正公平性缺口：另做24576参数/8973步/相同未标注窗口/源val选择的Mean Teacher，仅读取视觉输入清单、不读Qwen标签。
其真实interface及6步smoke通过（6次EMA更新），相关回归29项通过；旧Mean Teacher只保留历史参考。
提交1001832的四条方法队列已启动：每卡串行训练与独立完整推理，最多4物理卡；复用健康完成主训练，不启动重复训练。
GT-free数量诊断与离线归因分别执行，全部登记预测封存前不读取目标GT。弱缓存中的源margin是端点元数据，离线分析改用实际中间帧Frozen分数。

## 2026-10-30 PSR 最终闭环：三seed有整体收益，独立机制未建立，停止

- 本轮按用户明确“没有40直接用val”完成25Val同集无标签offline适应；不是40Train→Val、盲测、source-free online TTA。正式40Train匹配MOT仍0/40，不用GT轨迹或替换前端补造。
- 完整13学生（首轮8、seed0/42配对4、后验权重控制1）各8973步/24576参数，并各完成25视频425表达推理；47套官方评价及完整预测归档实际开包、425官方行和最终训练收据核验通过。包含复用基线/后验数量控制，不是47独立模型。
- 三seed Naive HOTA7.4499/7.5959/7.8064，PSR8.4204/8.8701/8.8763；均值样本标准差Naive7.6174±0.1792、PSR8.7223±0.2614，固定Frozen8.2180。真实点值提升存在，不能说“完全没有提升”；也不挑较好seed替代全表。
- 三seed PSR@Frozen数量8.0336/8.0312/8.0207均低于Frozen；Present HOTA8.8043/9.3266/9.3328仍低于9.7456，Empty FP显著下降。恢复GT净+748/+1925/+1917，但新增FP行18327/21261/21486，不能把删除输出/全局AUROC等同语言语义恢复。
- 已确认主代码soft BCE为最后decoder层Yes-No logit差，独立候选BCEWithLogits；未发现双sigmoid或错误时间轴等实现缺陷，不以此重跑。正确教师POS转移仅PSR38/41/41个（374），Naive37/39/37个（363）；BCE确降但固定正决策仍不足，未按GT改阈值/容量/步数。
- 必要POST_HOC端点重加权控制完成：263几何机会、526已有端点、0新标签、0新教师，HOTA8.4255；对Discovery+0.6128、对Full+0.0051且视频CI[−0.0118,+0.0251]。普通正例权重已重现扩展首轮表现，新中间帧与Full门均无独立收益证据；不是精确梯度干预或多seed等价性证明。
- 13完整学生训练+推理共24.4178 GPUh；继承教师/Frozen/source-only/前端、独立interface/smoke另列。共享GPU耗时不是独占利用率；无新GPU工作，未停止外部进程，已完成单元不重跑，失败入场/CPU锁等待均保留。
- 仅将首轮自动“需三seed确认”报告标签更新为最终归因结论，保留initial_automatic_verdict；未修改预测、官方指标或历史P1/旧UDA结果。最终科研判决POINT_GAIN_REPLICATED_BUT_INDEPENDENT_PSR_CONTRIBUTION_NOT_ESTABLISHED。
- 代码、全部正式预测/官方明细/成本/归因/日志与恢复索引逐阶段发布；原始图像、权重、raw教师库留服务器。所有队列正常退出；停止本轮，不自动训练第二域、重跑六TTA或堆新loss。
- 最终十文件相关CPU回归58项/5.15秒、compileall、diff检查通过；此前75项是不同重叠选择，不累加为133。真实GPU接口/smoke与13完整训练/推理另有实测收据，不用测试数字代替科研验收。

## 2026-10-31 D-FIT 归因预注册：官方同域覆盖冲突，按硬条件停止

- 从958e1d7新建research/v2dance-diag-attrib-20261031，先推送完整矩阵/校准/统计预注册0daa63a；未更改任何旧预测或训练。
- FROZEN/D1/D2/D3各425表达、3,588,445行margin核验通过，逐行候选/框/ID复现原生99,704/345,302/996,330/695,148输出。
- 官方Dance旧分数仅210,393/211,085条MOT，缺692条末观测（11,764候选–表达），已覆盖部分可复现旧254,118行；根因为vendor数据加载器range右端排他。
- 只在入场空闲GPU4补缺失logits，严格官方Dance权重、修正tokenizer、FP32、0优化步、0目标GT打开；8帧历史/原forward对照选择翻转0。
- 真实补齐产生792条新增正预测、影响56表达，完整原生行数254,910，与强制复现旧254,118不相容。按用户硬条件停止，不把新margin传入矩阵、不沿用旧HOTA32.134、不删新增正例强行通过。
- GPU补分数186.155秒/0.051710GPU-process小时，峰值3.977GiB；CPU输入审计51.533秒，3项针对性测试通过。一次裸python不可用的compile调用改用既定绝对解释器，保留报错；不是科研改进。
- 结果及停止原因见results/diag_attrib/REPORT.md；矩阵/基线/bootstrap明确NOT_RUN，A/B/C未验收。保留所有旧结果和新补分数，等待用户明确授权是否改用完整同域官方参照；不自行继续剩余实验。

### 2026-10-31 用户授权v2：完整同域参照对齐通过

- 先提交并推送PREREG_v2.md（公开555e8ef），新结果独立放results/diag_attrib/v2与diag_attrib_20261031_v2；v1及旧结果/vendor不改。
- CPU封存完整官方254,910行与所有margin，seal84f7e10b；原210,393观测上与旧254,118逐行一致，792新增正例均来自692末观测。
- 本轮新跑两次官方25视频425表达TrackEval：完整HOTA32.139/DetA28.354/AssA36.486/IDF1 27.816；历史覆盖32.134及254,118完全复现。之后所有主归因统一用完整参照，不把覆盖变化当TTA收益。
- 新GPU0、优化0、GT-free目标GT尝试0；完整参考CPU准备165.998秒，官方评价28.490/29.556秒。主矩阵/校准/视频CI后续按原预注册运行，不调整定义。
- 八项必要CPU回归通过；向量同帧排序与旧top-k tie规则一致，历史掩码与100帧因果窗口经针对性验证。不是以测试通过替代科研结果。

### 2026-10-31 D-FIT归因v2最终闭环：数量分配有学习信息，实用排序证据不足，停止

- 完成30格排序×数量、8个GT-free校准、7个统一历史覆盖敏感性、11个B0固定点；58个有效预测集各25视频425表达，共24,650个逐文件封存SHA。有效新官方TrackEval46套，逐行等价复用12；GT-free进程目标GT尝试0，新GPU/教师/优化0。
- 五个主对角与六个旧count-controls全425表达逐行复现；主官方32.139/254,910，历史官方32.134/254,118。完整GT仍212,541，40 Present/385 Empty，未改GT、框、ID、帧或vendor。
- 判定A“学生提供了超出简单校准的存在性信息”：最佳固定GT-free B1_GLOBAL HOTA13.848，对D1 29.702的Δ=15.8544，视频CI[9.9491,22.8811]；B0 Frozen全局同345,302行只有12.930，而Frozen@D1逐帧k=29.400，说明不是增加全局数量即可，k本身承载训练后的查询/时刻信息。
- D1−Frozen@D1=0.3025，CI[-0.4386,1.1631]，13视频正/12负；不能把HOTA非线性差值换算成因果贡献百分比，也不能把count-matched当免训练Frozen。
- 判定B“排序信号只在 oracle 数量下可见”：D3−Frozen在K=完整官方+0.0257，CI[-0.1160,0.1680]；K=D1 +0.6313，CI[-0.2294,1.7479]；K=GT +1.7041，CI[0.3447,3.3957]。不将R@1提升包装成实际可部署HOTA排序成功。
- 判定C：D1/完整官方1.3546倍在同量级；D2 3.9086、D3 2.7270过量；D1的81,061非空帧也多于官方41,979，Query-Empty仍156,758预测，存在性误发射未解决。D1−同域官方−2.4365，CI[-7.4624,2.5120]，不是等效/追平证明。
- 真实实现失败：B0绝对预算误用选中率截断，四个>1倍GT点被截212,541；未推送初版完整报告撤销并全部留档。按原预注册修为265,676/318,812/425,082/637,623，只重评四点；总实际新评价50次（有效46+无效4），失败成本/日志不隐藏，主矩阵与bootstrap修复前后SHA一致。
- 统一历史Frozen为当前封存margin子集8.2072/99,320，不偷换早期batch8.2060/99,316；重叠3,576,681分数审计仅6个近零符号差，净+4，逐候选差已披露，旧文件不改，主Frozen仍8.2180。
- 最终12项不同针对性测试通过，B0曲线已实际查看；旧1,700预测/margin对、3个final checkpoint、v1报告与预注册均复核未变。完整表、区间、GT审计、成本、原始官方CSV/日志、失败与恢复路径见results/diag_attrib/v2/REPORT.md及归档。
- 仅同集开发Val事后诊断，不是盲测、Train→Val、新方法或新的source-free online TTA；这些冻结学生历史上用过源监督与MLLM。所有本轮计算完成后停止，未启动额外训练、容量/pos_weight修改或下一阶段研究。

### 2026-10-31 D1 Train→Val：P0盘点与原始几何候选

- 从公开619390d新建research/v2dance-train2val-20261031，旧归因v1/v2与全部checkpoint不改。只读找到40Train原始DanceTrack数据入口，41,796图像/348,930六列几何行/419轨迹完整；不是之前缺失的Train预测NeuralSORT。
- Refer-Dance Train每视频39文本、共1,560组合。官方Val425均枚举本视频17表达，不跨视频取文本；Train/Val视频互斥，但8表达文件名重合，不能声称全表达未见。混合JSON仅独立进程解码sentence，label对象词法跳过。
- 按用户新授权把原始MOT六列原样转10列，逐行frame/id/box文本一致；共13,608,270候选–表达，数据在/data2/user/TTA-RMOT-assets/train2val/20261031，源数据不改。Train oracle几何/ID与Val预测前端差异明确披露。
- 准备训练分支的逐Train视频kernel白名单与早期输入审计，禁止Val图像/MOT/表达、旧Val教师库与D-FIT文件；尚未把未执行测试或实验记为成功。P0/P1新GPU成本0，后续按原D1三seed+D3一seed8973/K4/final_step完成，不调参。
- 24项相关测试通过；旧Naive8500全部425query/8500样本位置、ID、框、原因和分数由独立进程原样重放。隔离测试中kernel拦截全部5种输入，但audit漏记MOT文件stem，已定向修复并即时保存尝试计数；真实训练尚未开始，不能把故意测试次数当真实污染。
- GPU4/5/7/1共享准入完成40Train稀疏源分数，FP32原单query、strict V2权重、目标GT0；parent wall344.614秒，worker185.107/211.327/225.369/344.293秒。未重跑完整Frozen或训练跟踪器。
- 冻结Naive Train31200判断/7800窗口，manifest c09a1612；真实Qwen4端点smoke通过，26.128秒，BF16单帧、原P1后端和裁剪代码不改、原始logits有限、缓存可重开、冻结参数SHA相同、GT0。全Train教师下一步；8973×K4平均每窗口4.6015次曝光，不能为追平旧Val约16.89次而增加预算。
- 全40Train教师31200有效软判断完成，POS1765/31200=5.657051%；历史Naive656/8500（非PSR657）。全量4worker child秒149.462/162.007/167.927/308.882、parent310.098秒，新增31196 prompt+smoke缓存4，冻结权重/输入白名单/GT0通过。seal c3939408；实际split=train加载40/1560/7800/31200通过，后续expanded_kind=none/expanded0，不用兼容weak文件训练。
- 新训练保持D1三个seed和D3原配置逐字段数值相等；额外两个定向测试覆盖配方不变与预注册判定不新增规则。新的完整官方/归因脚本只绑定新输出目录和发布门，复用v2 TrackEval/数量排序/视频统计，不触碰旧结果。
- 两个新Train-only200步smoke正常0退出，原lr2e-4/无回退，D1参数24576/D3 2758656，目标梯度200/200非零，初始V2预测误差/翻转均0，冻结参数不变、D3 head更新、GT与禁止输入尝试0。完整训练必须从源V2重新初始化，不能把smoke权重当正式起点。
- 仅打包失败：copy2继承公开seal的0444权限，再次复制同SHA时报PermissionError。最小修复为复用相同SHA不可变元数据，新增只读文件回归；不chmod/修改封存cache，不重跑任何健康模型单元。
- 四臂正式8973/K4已从V2分别在GPU3/4/5/7启动（D1 seeds17/23/42、D3 seed17），最近已收到三D12000步/D31000步报告，原lr2e-4、无数值回退；这只是最近stdout报告点，不以loss或梯度代替完整结果。继续原blocking parent，不做agent进程/GPU轮询，不开Val。
- 在任何本轮Val预测/评价前，固定仅报告的文本覆盖补充：Train/Val整句小写/空白规范精确匹配，封存后按已见/未见句子复用官方CSV汇总；不改用户425序列验收规则。P0目录级25×17与40×39是同构全表达组合，并不保证每句在各视频有目标。新增匹配重分配FP差额，区别新增/删除行FP；官方CSV压缩公开，raw路径/SHA单列。
- 独立CPU确认跨归档Train图像对齐：40视频每个首/中/尾共120帧，原始JPG与Refer-Dance PNG解码RGB逐点相同、尺寸相同。完整数据帧数和原始GT帧号盘点此前已通过，本检查不读Val/referring GT；未改变健康训练或数据。
- CPU复用Train源margin与新教师标签，冻结初始拟合POS180/1765=10.1983%、非POS selected6.9407%、同帧teacher pair agreement0.521338；不是GT召回/准确率，不用于训练或选择。公开39表达/40视频分布。NEG统计桶含有效p<.95的UNKNOWN，优化仍raw-soft，不转硬负。模型曲线不得作GT成功证明。
- 三D1原lr2e-4正常完成8973步，父命令收到0退出（4245.369/4282.627/4245.507秒），40目标视频/1560组合/31200实际样本与17源视频覆盖，frozen不变/GT与禁止输入尝试0，final SHA ffc2574c/cb39097d/11ae8649核验。
- 真实Train拟合失败信号：教师POS选中率s17=.089518、s23=.086686、s42=.076487，低于Frozen .101983；非POS降至.00455/.00418/.00404，pair .589/.572/.578。固定300窗口曲线全程POS偏低，不能挑中间模型掩饰；source-dev CE .894/.901/.898无发散，不用非数值问题触发LR回退。还没有Val结果，原预算每窗口4.60（旧16.89）与候选来源差异使“只是记忆”不能单独因果归因。
- D3已收到8000步，继续原队列；先发布完成D1的Train记录，不开完整表A门、不提前Val，也不取消健康D3。G/std与判定使用已验证官方CSV全精度HOTA，只展示时舍入，主阈值仍固定13.848、公式仍8.218/29.702。
- D3随后原lr2e-4完成8973步/0退出，child5677.683秒、全队列5678.015秒；GT/禁止输入0/frozen不变/head更新/源计划SHA与旧D1相同，final SHA72c9eb38…。教师POS选中率.240793、非POS .038424、pair .649366、source-dev CE .675946；仍仅拟合部分教师正例，不能把BCE1.3118降至.3096当GT成功。
- 四臂完整表A生成，SHA412996980d43fd2a2dd37a339d4f240b6b8907a336454a1d68cccc925e8d9a78；step-final唯一checkpoint、未用目标GT选参或seed。source计划逐字节相同、目标40/1560/31200覆盖完整；最终27项不同CPU回归15.57秒通过。先推送并核验本表，再打开Val推理门；本阶段没有Val模型结果。
- 完整表A已公开be48a3b3a3c044d5a4f0679ca8a3c8d4dd0f9afc并回读核验同一tree；核验时evaluation目录不存在，发布门记录SHA不改。随后四final-step模型分别GPU0/2/3/4共享准入完整425表达推理，单blocking parent正常按视频封存；全部封存前不打开GT，不重启健康episode。
- P4四臂完整425表达推理正常退出，parent1959.932秒，GPU0/2/3/4的child1959.592/1789.587/1794.176/1828.297秒；每臂25视频、3588445候选–表达分数、最终step8973、测试时更新0、GT尝试0，MOT框/ID不改。D1输出s17=17040/s23=16053/s42=16901，D3=94563；尚不能仅凭缩量判断GT收益。
- 原生预测/margin SHA均核验；独立GT-free CPU进程复用原v2操作封存22套原生/参照/数量排序对照，目标GT尝试0。全部封存后才允许独立官方TrackEval；六套严格逐行等价的历史参照只复用原官方CSV，不重复推理或评测。到此所有新GPU进程总7.833500小时（含启动/CPU段的child elapsed，不是利用率积分），后续只有CPU评测/归因。
