let numbers = [0,0,0,0,0,0];
let time = 0;
// 顯示數字
function displayNumbers() {
    let numberContainer = document.getElementById('numberContainer');
    numberContainer.innerHTML = '';
    numbers.forEach(number => {
        let div = document.createElement('div');
        div.className = 'number';
        div.textContent = number;
        numberContainer.appendChild(div);
    });
}
// 產生隨機數字
function generateRandomNumber() {
    return Math.floor(Math.random() * 49) + 1;
}
// 更新數字
function updateNumbers() {
    if (numbers.some(element => element === 0)) {
		// 如果有 0，則找到第一個並替換為隨機數字
        let index = numbers.indexOf(0);
        numbers[index] = generateRandomNumber();
    } else {
		// 否則移除第一個數字，並將新的隨機數字添加到末尾
        numbers.shift();
        numbers.push(generateRandomNumber());
    }
}
// 設定計時器
setInterval(() => {
	if(time === 0){
		// 時間為 0 時顯示初始數字
		displayNumbers();
		time++;
	}else{
		// 在之後的每個時間間隔更新數字
		time++;
		updateNumbers();
		displayNumbers();
	}
// 每 1500 毫秒執行一次
}, 1500);
