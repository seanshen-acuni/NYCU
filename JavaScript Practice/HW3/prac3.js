//生成卡片裡面元素
    function generateNumbers(rule) {
        let numbers = [];
        for (let i = 0; i < 64; i++) {
            if (rule(i)) {
                numbers.push(i + 0);
            }
        }
        return numbers;
    }
	//設定規則
    function rule1(index) {
        return index % 2 === 1;
    }
    function rule2(index) {
        return index % 4 >= 2;
    }
    function rule3(index) {
        return index % 8 >= 4;
    }
    function rule4(index) {
        return index % 16 >= 8;
    }
    function rule5(index) {
        return index % 32 >= 16;
    }
    function rule6(index) {
        return index >= 32;
    }
	
	//定義每張卡片名稱和規則
    const chapters = [
        {name: "第1張卡片", rule: rule1},
        {name: "第2張卡片", rule: rule2},
        {name: "第3張卡片", rule: rule3},
        {name: "第4張卡片", rule: rule4},
        {name: "第5張卡片", rule: rule5},
        {name: "第6張卡片", rule: rule6}
    ];
	//生成卡片陣列
    function createCards() {
        for (let i = 0; i < 6; i++) {
            let card = document.createElement('div');
            card.className = 'card';
            let numbers = generateNumbers(chapters[i].rule);
            let numbersHTML = numbers.map(number => '<div class="number">' + number + '</div>').join('');
            let tickbox = '<input type="checkbox">';
			card.innerHTML = '<div>' + chapters[i].name + tickbox + '</div><div class="numbers">' + numbersHTML + '</div>';
			document.body.appendChild(card);
			if ((i+1) % 3 === 0) {
            document.body.appendChild(document.createElement('br'));
		}	
        }
    }
    createCards();
