<template>
  <div>
  <div class="hospital box container">
    <div class="hospital_tittle">
      <i>请选择你要联合诊断的医院</i><br />
      <em>接入更多医院的数据库，您的诊断结果会更加准确</em>
    </div>
    <div class="hospital_content">
      <div class="chose_hos box">
        <div class="contentdetail">
            <img src="../assets/img/jnu_hospital.jpg" alt="暨南大学附属第一医院" >
            暨南大学附属第一医院
        </div>
        <div style="flex: 1; text-align: right">
          <label>
            <input type="checkbox" v-model="which" value="jack" /> 选择
          </label>
        </div>
      </div>
      <div class="chose_hos box">
        <div class="contentdetail">
            <img src="../assets/img/hos2.jpg" alt="暨南大学附属第一医院" >
            中山大学附属第一医院
        </div>
        <div style="flex: 1; text-align: right">
          <label>
            <input type="checkbox" v-model="which" value="jack1" /> 选择
          </label>
        </div>
      </div>
      <div class="chose_hos box">
        <div class="contentdetail">
            <img src="../assets/img/hos_nan.jpg" alt="暨南大学附属第一医院" >
            南方医科大学南方医院
        </div>
        <div style="flex: 1; text-align: right">
          <label>
            <input type="checkbox" v-model="which" value="jack2" /> 选择
          </label>
        </div>
      </div>
      <div class="chose_hos box">
        <div class="contentdetail">
            <img src="../assets/img/hos_sheng.jpg" alt="暨南大学附属第一医院" >
            广东省人民医院
        </div>
        <div style="flex: 1; text-align: right">
          <label>
            <input type="checkbox" v-model="which" value="jac3k" /> 选择
          </label>
        </div>
      </div>
      <div class="chose_hos box">
        <div class="contentdetail">
            <img src="../assets/img/hospital_1.png" alt="暨南大学附属第一医院" >
            广州市第一人民医院
        </div>
        <div style="flex: 1; text-align: right">
          <label>
            <input type="checkbox" v-model="which" value="jack4" /> 选择
          </label>
        </div>
      </div>
      <div class="chose_hos box">
        <div class="contentdetail">
            <img src="../assets/img/hospital_medicine.jpg" alt="暨南大学附属第一医院" >
            广州中医药大学第一附属医院
        </div>
        <div style="flex: 1; text-align: right">
          <label>
            <input type="checkbox" v-model="which" value="jack5" /> 选择
          </label>
        </div>
      </div>
    </div>
   
  </div>
  <div class="illness_box">
    <div class="hospital_tittle">
      <i>请选择你要联合诊断的病症类型/科室</i><br/>
      <select class="illness_choose" v-model="selectedIllness">
            <option value="">请选择一个选项</option>
            <option v-for="illness in illnesses" >{{illness.name}}</option>
        </select>
        <div>
          <button @click="confirmSelection" :disabled="isConfirmButtonDisabled || !atLeastOneHospitalSelected" :style="buttonStyle"  style="margin-top:30px;width: 100px;margin-left:20px;background-color: #DCDCDC;color:black;border-color: black;border-radius: 10%;">确定</button>
        </div>
        <div v-if="showProgress" class="progress-container">
          <div class="progress-bar" :style="{ width: progress + '%' }"></div>
          <div class="progress-text">{{ progressText }}</div>
        </div>
      </div>
  </div>

  <div class="container" v-if="progress >= 100">
    <div class="content-title">上传图片</div>
    <!-- action :上传服务器接收地址 -->
    <el-upload
      class="upload-demo"
      drag
      action="http://127.0.0.1:5000/upload"
      :on-change="uploadImageForDiagnosis"
    >
      <el-icon class="el-icon-upload"><upload-filled /></el-icon>
      <div class="el-upload__text">
        将文件拖到此处，或
        <em>点击上传</em>
      </div>
    </el-upload>
      <div> 
        <button @click="viewDiagnosisResults" :disabled="!diagnosisImages.length || activeButton === 'results'" :class="{ active: activeButton === 'results' }" style="margin-left:20px;">查看诊断结果</button>
      </div>
      <div v-if="showDiagnosisResults" class="result-viewer">
        <div v-if="!hasStartedShowingResults" class="loading-message">
        </div>
        <div>
          <img :src="currentImageUrl" alt="Selected Image" />
          <div class="image-buttons">
            <button @click="previousImage" :disabled="currentImageIndex === 0" :class="{ active: activeButton === 'previous' }">&lt;</button>
            <button @click="nextImage" :disabled="currentImageIndex === diagnosisImages.length - 1" :class="{ active: activeButton === 'next' }">&gt;</button>
          </div>
        </div>
    </div>
  </div>
</div>
</template>

<script>
export default {
  name: "Common",
  data() {
    return {
      isCollapse: false, // 不收缩
      asidWidth: "200px", // 侧边栏宽度
      user: localStorage.getItem("user") ? JSON.parse(localStorage.getItem("user")) : {},
      diagnosisImages: [], // 存储诊断结果的图片
      showDiagnosisResults: false, // 是否显示诊断结果
      currentImageIndex: 0, // 当前显示的图片索引
      isUploading: false, // 控制上传按钮的高亮
      activeButton: null, // 控制当前激活的按钮
      hasStartedShowingResults: false, // 是否已经开始显示诊断结果
      showLoadingMessage: true, // 控制是否显示加载中的提示信息
      fileInputRef: null, // 用于引用文件输入元素
      which:[],//用于存储选中的医院
      canUpload: false, // 控制是否可以上传图片
      illnesses:[{name:'肺病'},{name:'肝病'},{name:'肾病'},{name:'脾病'},{name:'胰腺病'},{name:'等等'}],
      selectedIllness:null,
      showProgress: false,
      progress: 0,
      progressText:'联邦诊断模型训练中',
      buttonStyle: null, // 控制按钮样式
      isConfirmButtonDisabled: false, // 控制确认按钮是否禁用
    };
  },
  computed: {
    // 新增计算属性，检查是否有至少一个医院被选中
    atLeastOneHospitalSelected() {
      return this.which.length > 0;
    },
    loginStatusText() {
      return this.user.username ? '退出登录' : '登录';
    },
    currentImageUrl() {
      return this.diagnosisImages[this.currentImageIndex] || '';
    }
  },
  methods: {
    // 更新确认按钮的点击事件，如果至少选择了一个医院，则允许点击
    confirmSelection() {
      if (this.atLeastOneHospitalSelected && this.selectedIllness) {
        // 用户选择了至少一个医院，允许上传图片
        this.canUpload = true;
        this.buttonStyle = { backgroundColor: '#FF5722', color: 'white' };
        this.showProgress = true; // 显示进度条
        this.setProgress(); // 开始更新进度
        this.isConfirmButtonDisabled = true;
        // 执行确认选择的逻辑
        // console.log('至少选择了一个医院', this.which);
      } else {
        // 提示用户至少选择一个医院
        alert('请至少选择一个医院');
      }
    },
    setProgress() {
      let currentProgress = 0;
      const interval = setInterval(() => {
        currentProgress += 5; // 每次增加 3.33%，大约 30 秒完成
        this.progress = currentProgress;
        if (currentProgress < 100) {
          this.progressText = '联邦诊断模型训练中'; // 加载过程中的状态文本
        } else {
          this.progressText = '模型训练完成，请上传图片'; // 加载完成后的状态文本
          clearInterval(interval);
        }
      }, 1000); // 每秒更新一次
    },
    updateIsCollapse() {
      this.isCollapse = !this.isCollapse;
      this.asidWidth = this.isCollapse ? "64px" : "200px";
    },
    navigateToLoginOrLogout() {
      if (this.user.username) {
        this.logout();
      } else {
        this.$router.push('/login');
      }
    },
    logout() {
      localStorage.removeItem('user');
      this.$router.push('/login');
    },

    handleUploadChange(file, fileList) {
      // 处理文件上传逻辑
      console.log('文件已选择:', file, fileList);
      // 可以在这里调用上传逻辑
    },
  
    uploadImageForDiagnosis(file, fileList) {
    this.activeButton = 'upload';
    this.isUploading = true;

    // 使用 fileList 中的第一个文件进行上传
    const fileToUpload = fileList[0].raw; // 获取原始文件对象

    const formData = new FormData();
    formData.append('file', fileToUpload);

    fetch('http://127.0.0.1:5000/upload', {
      method: 'POST',
      body: formData
    })
    .then(response => {
      if (!response.ok) {
        throw new Error(`HTTP error! Status: ${response.status}`);
      }
      return response.blob(); // 获取 Blob 对象
    })
    .then(blob => {
      const imageUrl = URL.createObjectURL(blob); // 创建一个 URL 对象指向 Blob 数据
      this.diagnosisImages.push(imageUrl); // 将图像 URL 添加到数组中
      this.isUploading = false;
      this.activeButton = null;
    })
    .catch((error) => {
      console.error('Error:', error);
      this.isUploading = false;
      this.activeButton = null;
    });
  },
    
    viewDiagnosisResults() {
      this.showDiagnosisResults = true;
      console.log('查看诊断结果:', this.diagnosisImages); // 调试输出

      // 设置当前激活的按钮
      this.activeButton = 'results';

      // 如果是第一次显示诊断结果
      if (!this.hasStartedShowingResults) {
        this.hasStartedShowingResults = true;
        this.showLoadingMessage = true; // 显示加载中的提示信息

        // 弹出提示信息
        // alert('正在与医院进行沟通，请耐心等待多家医院的联合诊断结果...');

        // 延迟一段时间后再显示图片
        // setTimeout(() => {
        //   this.showLoadingMessage = false; // 关闭加载中的提示信息
        //   console.log('延迟3秒后设置 hasStartedShowingResults 为 true');
        // }, 3000); // 模拟延迟3秒
      }
    },
    previousImage() {
      if (this.currentImageIndex > 0) {
        this.currentImageIndex--;
        this.activeButton = 'previous';
      }
    },
    nextImage() {
      if (this.currentImageIndex < this.diagnosisImages.length - 1) {
        this.currentImageIndex++;
        this.activeButton = 'next';
      }
    },
    loadDiagnosisImagesFromServer() {
      // 模拟从服务器加载诊断图像
      this.diagnosisImages = [
        'https://example.com/path/to/diagnosis_image1.jpg',
        'https://example.com/path/to/diagnosis_image2.jpg',
        'https://example.com/path/to/diagnosis_image3.jpg'
      ];
    }
  },
  mounted() {
    // 创建一个文件输入元素引用
    this.fileInputRef = document.createElement('input');
    this.fileInputRef.type = 'file';
    document.body.appendChild(this.fileInputRef);
  }
};
</script>

<style scoped>


.el-main {
  padding: 10px;
}

button {
  background-color: #4CAF50; /* Green */
  border: none;
  color: white;
  padding: 15px 32px;
  text-align: center;
  text-decoration: none;
  display: inline-block;
  font-size: 16px;
  margin: 4px 2px;
  cursor: pointer;
}

button:disabled {
  background-color: #ccc;
  cursor: not-allowed;
}

.progress-container {
  margin: auto;
  margin-top: 20px;
  width: 90%;
  background-color: #eee;
  border-radius: 5px;
  overflow: hidden;
}

.progress-bar {
  height: 20px;
  background-color: #4CAF50;
  width: 0%;
  border-radius: 5px;
  transition: width 0.5s ease-in-out;
}

.progress-text {
  position: absolute;
  width: 80%;
  text-align: center;
  line-height: 20px; /* 与进度条高度相同 */
  color: #000;
}

button.active {
  background-color: #FF5722; /* 激活时的颜色 */
}

img {
  max-width: 100%;
  height: auto;
  display: block; /* 确保图片不会有额外的空隙 */
  margin: 0 auto; /* 居中图片 */
}

.loading-message {
  text-align: center;
  padding: 20px;
  color: #4CAF50;
  font-weight: bold;
}

.result-viewer {
  display: flex;
  flex-direction: column;
  align-items: center; /* 居中对齐 */
  justify-content: center; /* 居中对齐 */
  text-align: center;
}

.image-buttons {
  display: flex;
  justify-content: center; /* 居中对齐 */
  margin-top: 10px; /* 图片下方的按钮间距 */
}

.image-buttons button {
  margin: 0 5px; /* 按钮之间的间距 */
}

.container{
  background-color: #fff;
}

.content-title {
  font-weight: 400;
  line-height: 100px;
  margin-top: 30px;
  margin-left: 30px;
  font-size: 22px;
}

.upload-demo {
  width: 360px;
  background-color: #fff;
  margin-left: 30px;
}
.hospital {
  height: 300px;
 font-family: cursive;
 background-color: #fff;
}

.illness_box{
  margin-top: 10px;
  height: 250px;
  background-color: #fff;
}

.illness_choose{
  margin: auto;
  margin-top: 10px;
  height: 30px;
  width: 300px;
  margin-right: 100px;
}

.hospital_tittle {
  flex: 1;
  padding-top: 10px;
  margin-left: 20px;
  font-family: cursive;
}
.hospital_tittle em {
  font-size: 15px;
  font-family: cursive;
  color: rgb(108, 108, 118);
}
.hospital_tittle i {
  font-size: 30px;
  color: rgb(31, 31, 88);
}
.hospital_content {
  flex: 4;
  display: flex;
  justify-content: space-around;
  align-items: center;
}

.chose_hos {
  border: solid 3px rgb(28, 113, 122);
  border-radius: 5%;
  height: 200px;
  width: 170px;
}
.box {
  display: flex;
  flex-direction: column;
}
.chose_hos img{
    height: 120px;
    width: 120px;
} 
.contentdetail{
    flex: 5;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
}
</style>
