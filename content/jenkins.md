+++
date = '2025-08-31T12:57:45+08:00'
draft = false
title = 'Jenkins'
tags = ["jenkins","CICD"]
categories = ["tools"]
+++


## tutorial

1. Pipline & jenkinsfile 手册：https://www.jenkins.io/doc/book/pipeline/getting-started/

2. 提供的全局变量 `env.`: `<Jenkins master的地址>/pipeline-syntax/globals#env`

3. pipline 实例：https://www.jenkins.io/doc/pipeline/examples/


## 尽可能应用Jenkins提供的命令，而不是一股脑儿使用shell脚本

比如，git clone，使用：

~~~groovy
checkout scmGit(branches: [[name: params.BRANCH]],
                extensions: [],
                userRemoteConfigs: [[credentialsId: params.GITHUB_CREDENTIAL, url: 'https://github.com/xxx.git']])
~~~

## jenkins是面向过程的，对于任务配置，多考虑使用表驱动

## Pipline 中访问 env 变量

~~~groovy
stage(){
    println "WORKSPACE: ${env.WORKSPACE}"
    echo env.WORKSPACE
    echo WORKSPACE
}
~~~

环境变量可以通过 Groovy 代码访问，方式为 `env.VARNAME` 或者直接使用 `VARNAME`。你也可以修改这些属性，但只能通过使用 `env.` 前缀来写入。所以：

1. jenkins job 中有保留的 env 变量，避免疑惑这些变量要加上 `env.`。
2. env 变量即使在机器上设定了，也是可以修改的。


## groovy中的数学计算

~~~groovy
def number = 243  // 要计算的数
def fifthRoot = number ** (1.0/5)  // 计算5次方根
~~~

## 两种pipline

在Jenkins中，有两种主要类型的Pipeline：Scripted Pipeline 和 Declarative Pipeline。

1. Scripted Pipeline【我的工作中都是这种的脚本】:

  - 使用Groovy语法编写，允许更灵活的流程控制和自定义逻辑。
  - 通过node和stage等关键字来定义流水线的执行节点和阶段。
  - 可以直接编写Groovy脚本来构建流水线。

2. Declarative Pipeline:

  - 使用更结构化的语法，更易于阅读和维护。
  - 通过pipeline、agent、stages等关键字来定义流水线的结构和执行环境。
  - 提供了更丰富的语法来定义构建、部署和测试等阶段。


## jenkins 中如何在多个 NODE 并行执行任务

~~~groovy
def parallel_tasks = [:]
if (params.GPU_TASK) {
    parallel_tasks["GPU_TASK"] = {
        node(params.GPU_NODE) {
            stage(){}
            ...
        }
    }
}
if (params.CPU_TASK) {
    parallel_tasks["CPU_TASK"] = {
        node(params.CPU_NODE) {
            stage(){}
            ...
        }
    }
}

parallel(parallel_tasks)
~~~

## Elvis 操作符 `?:`

这两句句话有什么不同：

~~~groovy
Apple = Banana ?: ""
Apple = Banana ? Banana : ""
~~~

两者操作符不同，`?:` 是Elvis操作符，表示如果Banana不是null或空，则将其值赋给Apple，否则Apple将会是空字符串。两者效果相同。

## 关于proxy

jenkins服务中写死的 proxy 会影响 groovy 的函数，但是在sh中的执行 你依然可以使用自己的proxy。


## 使用Groovy语法创建并写入Python脚本

~~~groovy
// Python脚本的内容
def pythonScriptContent = '''
    print("Hello from Jenkins Pipeline!")
    # Add more Python code here
'''
// 将内容写入Python脚本文件
writeFile file: 'script.py', text: pythonScriptContent
// 执行Python脚本
sh 'python script.py'
~~~

~~~groovy
def createPythonScript() {
    def scriptName = sh(script: 'echo \'print("Hello, world!")\' > example.py', returnStdout: true).trim()
    return scriptName
}
node {
    stage('Create Python Script') {
        def scriptName = createPythonScript()
        echo "Created Python script: ${scriptName}"
    }
    stage('Other Stage') {
        def scriptName = createPythonScript()
        // 在这里可以使用创建的Python脚本
    }
}
~~~


## jenkins 文件归档

如果在Jenkins上使用`archiveArtifacts`将生成的文件归档，即使在归档后删除了该文件，你仍然可以从Jenkins上访问该log文件，这是因为Jenkins将文件存档在特定的位置，而不是直接引用原始文件。当您使用 archiveArtifacts 步骤归档文件时，Jenkins将文件**复制**到其构建存档目录中，通常是Jenkins服务器上的特定位置。

通过页面的"Artifacts"或类似的部分，通常会提供一个链接或按钮，使您可以直接下载或查看归档的文件。注意，归档的文件会**占用磁盘空间**，因此建议在需要时删除归档的文件，以便释放磁盘空间。您可以在Jenkins的配置中调整构建存档的策略和保留期限，以满足您的需求。

~~~groovy
csv += ... // 构建CSV文件内容
writeFile file: "result.csv", text: csv
archiveArtifacts 'result.csv'
~~~

~~~groovy
writeFile file: 'run_ut.sh', text: '''
        #!/bin/bash
        set -ex
        export CI_CACHE=/ci_cache
        export SC_HOME=/AI
        export CI_HOME=/AI/3rdparty/AI-ci
        bash $CI_HOME/run_gcc_latest.sh
    '''
withCredentials([gitUsernamePassword(credentialsId: params.GITHUB_CREDENTIAL, gitToolName: 'Default')]) {
    sh "bash scripts/ci/checkout.sh"
}
...
sh """ sh ./run_ut.sh """ // 上文创建文件，这里执行
~~~


## 访问 已经归档的 文件，访问已有的 job 已有的 build

~~~groovy
copyArtifacts filter: '*',
        fingerprintArtifacts: true,
        projectName: 'Triton_Release',
        selector: specific(env.IPEX_BUILD_NUMBER)
~~~

~~~groovy
import hudson.model.*
import jenkins.model.*

def jobName = "my-job" // 作业名称
def buildNumber = 42 // 构建号

def job = Jenkins.instance.getItem(jobName)
def build = job.getBuildByNumber(buildNumber)

if (build != null) {
    def artifacts = build.artifacts
    artifacts.each { artifact ->
        def fileName = artifact.fileName
        def relativePath = artifact.relativePath

        println("File: $fileName, Path: $relativePath")
    }
} else {
    println("Build $buildNumber not found for job $jobName.")
}
~~~

## groovy中使用 `.each` 循环 遍历一个 map，在遍历中进行 if 判断

~~~groovy
data.each { model, modes ->
    println(model + ":")
    modes.each { mode, types ->
        println("\t" + mode + ":")
        types.each { type ->
            if (type == 'int8') {
                println("\t\t" + type + " (optimized)")
            } else {
                println("\t\t" + type)
            }
        }
    }
}
~~~

## publich HTML report

~~~groovy
dir('result') {
    copyArtifacts filter: '*_perf.log', fingerprintArtifacts: true,
                    projectName: 'Complex_fusion',
                    selector: specific(env.RESULT_NUMBER)

    sh "touch conv_block.log mha.log mlp.log misc.log"
}
dir('baseline') {
    copyArtifacts filter: '*_perf.log', fingerprintArtifacts: true,
                    projectName: 'Complex_fusion',
                    selector: specific(env.BASELINE_NUMBER)

    sh "touch conv_block.log mha.log mlp.log misc.log"
}
// generate report for each model
for (workload in ["conv_block", "mha", "mlp", "misc"]) {
    def accumulate_min = 1;
    def count_min = 0;
    def accumulate_avg = 1;
    def count_avg = 0;
    def summary = "<html><body>";
    summary += "<link rel='stylesheet' href='perf_test.css'>";
    summary += "<table border='1'>\n";
    summary += "<tr>" +
                "<td class='cases'>Case</td>" +
                "<td class='column'>Result Min</td>" +
                "<td class='column'>Result Avg</td>" +
                "<td class='column'>Baseline Min</td>" +
                "<td class='column'>Baseline Avg</td>" +
                "<td class='column'>Ratio Min</td>" +
                "<td class='column'>Ratio Avg</td>" +
                "</tr>\n";
    // make sure the log file exists
    sh "touch result/${workload}.log baseline/${workload}.log"

    def res = parse_result("result/${workload}.log");
    def base = parse_result("baseline/${workload}.log");

    // loop each entry in the result
    // line [case,min,avg] => result {case: [min,avg]}
    for (entry in res) {
        def case_name = entry.key;
        def perf_min = entry.value[0];
        def perf_avg = entry.value[1];
        def base_perf_min = 0.0;
        def base_perf_avg = 0.0;
        // find the base
        if (base.containsKey(case_name)) {
            base_perf_min = base[case_name][0];
            base_perf_avg = base[case_name][1];
        }

        def ratio_min = "/";
        def css_class_min = "na";
        if (base_perf_min != 0) {
            def res0 = compare(perf_min, base_perf_min, ratio_min);
                css_class_min = res0[0]
                ratio_min = res0[1]
        }
        if (ratio_min != "/" && ratio_min != "0.0") {
            accumulate_min *= ratio_min.toDouble();
            count_min += 1;
        }

        def ratio_avg = "/";
        def css_class_avg = "na";
        if (base_perf_avg != 0) {
            def res1 = compare(perf_avg, base_perf_avg, ratio_avg);
            css_class_avg = res1[0]
            ratio_avg = res1[1]
        }
        if (ratio_avg != "/" && ratio_avg != "0.0") {
            accumulate_avg *= ratio_avg.toDouble();
            count_avg += 1;
        }

        summary += "<tr>" +
                    "<td>${case_name}</td>" +
                    "<td>${perf_min}</td>" +
                    "<td>${perf_avg}</td>" +
                    "<td class='value'>${base_perf_min}</td>" +
                    "<td class='value'>${base_perf_avg}</td>" +
                    "<td class='value ${css_class_min}'>${ratio_min}</td>" +
                    "<td class='value ${css_class_avg}'>${ratio_avg}</td>" +
                    "</tr>\n";
    }
    def geomean_min
    def geomean_avg
    if (count_min > 0) {geomean_min = accumulate_min ** (1.0/count_min);}
    if (count_avg > 0) {geomean_avg = accumulate_avg ** (1.0/count_avg);}
    def css_geo_min = up_or_down(geomean_min)
    def css_geo_avg = up_or_down(geomean_avg)

    summary += "<tr>" +
                "<td class='geomean'>Geomean</td>" +
                "<td class='column'></td>" +
                "<td class='column'></td>" +
                "<td class='column'></td>" +
                "<td class='column'></td>" +
                "<td class='column ${css_geo_min}'>${geomean_min}</td>" +
                "<td class='column ${css_geo_avg}'>${geomean_avg}</td>" +
                "</tr>\n";
    summary += "</table>\n";
    summary += "</body></html>\n";

    dir('summary') {
        // 生成两个文件，html（内容）和css（格式），
        writeFile file: "perf_test.css", text: "" +
                    ".upgrade {background: lightgreen;}\n" +
                    ".downgrade {background: pink;}\n" +
                    ".na {background: lightgrey;}\n" +
                    ".value {text-align: right;}\n" +
                    ".cases {width: 500px;}\n" +
                    ".geomean {border: #333333; background: rgb(172, 206, 240);}" +
                    ".column {width: 100px;}\n";
        writeFile file: "${workload}.html", text: summary
    }
}
publishHTML([allowMissing: true,
            alwaysLinkToLastBuild: true,
            keepAll: true,
            reportDir: 'summary',
            reportFiles: 'conv_perf.html,mha.html,mlp.html,misc.html',
            reportName: 'PERF Report',
            reportTitles: '',
            useWrapperFileDirectly: true])
~~~


## Jenkins 脚本一般结构

~~~groovy
import org.jenkinsci.plugins.pipeline.modeldefinition.Utils
node(params.NODE) {

    def conda_env = pwd() + "/conda_llm";
    withEnv([
        'HTTP_PROXY=http://xxx.com:912',
        'HTTPS_PROXY=http://xxx.com:912',
        "CMAKE_PREFIX_PATH=${conda_env}",
        "MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000",
        "_CONSTANT_CACHE=1",
        "_GRAPH_CONSTANT_TENSOR_CACHE_CAPACITY=cpu:-1",
        "_DISABLE_COMPILER_BACKEND=${params.DISABLE_COMPILER}",
        "VERBOSE=${param.VERBOSE}"
    ]) {
        stage('setup & run') {
            deleteDir();
            dir('llm') {
                checkout scmGit(branches: [[name: params.BRANCH]],
                                extensions: [],
                                userRemoteConfigs: [[credentialsId: params.GITHUB_CREDENTIAL, url: 'https://github.com/xxx.git']])
                copyArtifacts filter: '*',
                        fingerprintArtifacts: true,
                        projectName: 'YYY_Release',
                        selector: specific(env.IPEX_BUILD_NUMBER)


                retry(5) { sh "conda create --prefix ${conda_env} --file requirements.txt -c conda-forge python=3.10 -y" }
                retry(5) { sh "conda install --prefix ${conda_env} jemalloc -y" }

                dir('examples/cpu/inference/python/llm'){
                    sh """
                        bash ./tools/env_setup.sh 7
                        source ./tools/env_activate.sh
                        source activate ${conda_env}
                        python run.py
                    """
                }
            }

            sh """
                source activate ${conda_env}
                ...
                pip install ${params.PYTORCH_URL} --no-deps
                pip install *.whl --no-deps
                pip install -r requirements.txt
            """
        }
        withEnv([
            "MALLOC_CONF=oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000",
            "_CONSTANT_CACHE=1",
            "GRAPH_CONSTANT_TENSOR_CACHE_CAPACITY=cpu:-1",
            "_DISABLE_COMPILER_BACKEND=${params.DISABLE_COMPILER}",
            "VERBOSE=${param.VERBOSE}"
        ]) {
            stage('run llm') {
                dir('vision') {
                    checkout scmGit(branches: [[name: 'master']],
                                    extensions: [],
                                    userRemoteConfigs: [[credentialsId: params.GITHUB_CREDENTIAL, url: 'https://github.com/xxx.git']])
                    sh """
                        source activate ${conda_env}
                        ...
                    """
                    for (dt in ["int8_ipex"]) {
                        sh """
                            source activate ${conda_env}
                            bash run_test.sh all ${dt} ${params.mode} int8_ipex
                        """
                    }
                    sh """
                        cp logs/summary.log ./vision.log
                    """
                    archiveArtifacts 'vision.log'
                }
            }
        }
    }
}
~~~


## 坑：shell的带特殊符号的环境变量

~~~groovy
withEnv(["CACHE_CAPACITY=cpu:10240;gpu:20480",
            "CACHE_CAPACITY_2='cpu:10240;gpu:20480'",
            "CACHE_CAPACITY_3=\"cpu:10240;gpu:20480\""]){
    echo env.CACHE_CAPACITY;       // cpu:10240;gpu:20480     v
    echo env.CACHE_CAPACITY_2;     // 'cpu:10240;gpu:20480'   x
    echo env.CACHE_CAPACITY_3;     // "cpu:10240;gpu:20480"   x
    sh"""
        export CACHE_CAPACITY_4="cpu:10240;gpu:20480"
        echo \${CACHE_CAPACITY}    ## cpu:10240;gpu:20480     v
        echo \${CACHE_CAPACITY_2}  ## 'cpu:10240;gpu:20480'   x
        echo \${CACHE_CAPACITY_3}  ## "cpu:10240;gpu:20480"   x
        echo \${CACHE_CAPACITY_4}  ## cpu:10240;gpu:20480     v
    """
}
~~~

`withEnv` 中变量的值，不加双引号！

## 坑：jenkins 中的用户访问不到 GPU

在机器上给用户访问权限后，直接在机器上是可以访问，但是jenkins的pipline中相同的用户还是不能访问。

答：jenkins系统上需要重启机器，后就可以了
    jenkins上的agent其实是一直运行 的，可以理解成jenkins系统中这个shell从来没有变过，重启才生效

## 坑：'' vs ""

在Groovy中，'' 和 "" 都表示空字符串，它们在功能上是等价的。因此，在`my_list.add('')`和`my_list.add("")`这两行代码中，它们的作用是相同的，都是向`my_list`中添加一个空字符串。 Groovy允许使用单引号或双引号来表示字符串，这两种表示方法在大多数情况下是可以互换使用的。

## 坑：withEnv()的参数是一个List，其内容可以是空但必须要给一个List 对象

jenkins 报错：hudson.remoting.ProxyException: java.lang.NullPointerException: Cannot invoke "java.util.List.iterator()" because "overrides" is null


## 坑：withEnv([])

jenkins 报错：process apparently never started in ...
答：检查withEnv中的环境变量是否符合预期。注意要在sh中检查环境变量的值，最好将所有的 env 输出到文件并且archive起来，便于检查

## 坑：shell 的 pid

不同的 sh """ """, 之间的 **pid 不相同**，不共用环境，包括conda环境。**如果要公用，则需要通过 withEnv添加**

## 坑：sh """ vd sh '''

sh """ 中的变量要是 jenkins 脚本的变量， **先获取变量值，替换后，再执行shell**
sh ''' 中变量是 shell 脚本自身的变量， **直接执行shell**

## 坑：案例分析

~~~groovy
sh (script: """
    source ~/miniconda3/etc/profile.d/conda.sh
    if [ "\$(conda info --env | grep ${CONDA_ENV} | wc -l )" == "0" ];then
        conda create -n ${CONDA_ENV} python=3.10 openpyxl csv -y
    fi
    conda activate ${CONDA_ENV}
    bash ${WORKSPACE}/tools/bdnn/jenkins_utils/collect_result.sh ${cfg[2]}_perf.log perf.log
    python ${WORKSPACE}/tools/bdnn/jenkins_utils/to_excel.py perf.log ${cfg[2]}_perf.xlsx
""", returnStdout: true)
~~~

if 条件中 `"\$(conda info --env..."`  的第一个`$`, 需要转义，告诉 jenkins 不要替换 `$`后的内容，其他jenkins变量替换之后，这个`$`作为shell脚本的内容执行。

## 坑：访问函数参数  sh""" vs  sh''' 两者区别

~~~groovy
def Select_Model(direction, dtype) {
    println("======inside of Select()=====>" + direction)
    sh'''
        echo "------ > $direction"
    '''
}
~~~

上面函数中变量`direction` 为什么在sh中访问不到，echo得到是空？

改为一下就可以：
~~~groovy
def Select_Model(direction, dtype) {
    println("======inside of Select()=====>" + direction)
    sh"""
        echo "------ > ${direction}"  // allows for string interpolation, 先获取direction值，后执行shell
    """
}
~~~

因为 `sh"""` 里通过 `$ `来获取groovy的变量，如上例，`$direction` 获取函数参数的值替换后，执行shell。而`sh'''` 里边不能获取groovy变量的值，只能通过 `withEnv` 的方式间接获取，所以如果要使用 `sh'''` 那么应改为：

~~~groovy
def Select_Model(direction, dtype) {
    withEnv(["Direction=${direction}"]) {
        println("======inside of Select()=====>" + direction)
        sh'''
        echo "------ > ${Direction}"  // 不会string interpolation, 原样输出，后执行shell，此时Direction已经是个环境变量了
        '''
    }
}
~~~


## 坑：dir ()

在Jenkins中，`dir('${MY}')`[错误] 和 `dir("${MY}")`[正确] 之间的主要区别在于引号的类型。单引号和双引号在Groovy中有不同的用途：

`dir('${MY}')` 使用了单引号，这意味着${MY}不会被解释为变量，而会被视为普通的字符串。这将导致Jenkins尝试切换到一个名为 `${MY}` 的目录，而不是切换到环境变量 MY 的值所表示的目录。因此，这种写法通常不会达到预期的效果，除非您确实有一个名为${MY}的目录。

`dir("${MY}")` 使用了双引号，这意味着${MY}会被解释为变量，并用其值替换。如果环境变量 MY 的值是一个有效的目录路径，那么Jenkins会尝试切换到这个目录。

所以，一般情况下，如果您希望在dir命令中引用一个环境变量的值作为目录路径，应该使用双引号，如`dir("${MY}")`。这将使Jenkins正确地解释${MY}作为变量，并使用它的值作为目录路径

所以，Jenkins中没有特殊理由，都是用双引号 """ 。


## 坑：方法不可用

默认情况下，许多敏感方法是被禁止的，以确保Jenkins的安全性。当您在Jenkins中执行Groovy脚本时，可能会遇到安全限制，例如 "Scripts not permitted to use method" 的错误消息。这是由于Jenkins的脚本安全性设置所引起的。

Administer Jenkins 权限的用户可以配置 Jenkins 的脚本安全性设置，以允许或拒绝特定方法的使用。
