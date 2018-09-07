const maze = "111111111111111111111111111111111111\n\
        1000000000000000000000000000000000S1\n\
        101111111111111111111111101111111101\n\
        101100010001000000111111100011000001\n\
        101101010101011110111111111011011111\n\
        101101010101000000000000011011000001\n\
        101101010101010111100111000011111101\n\
        101001010100010000110111111110000001\n\
        101101010111111110110000000011011111\n\
        101101000110000000111111111011000001\n\
        100001111110111111100000011011111101\n\
        111111000000100000001111011010000001\n\
        100000011111101111101000011011011111\n\
        101111110000001000000011111011000001\n\
        100000000111111011111111111011001101\n\
        111111111100000000000000000011111101\n\
        1E0000000001111111111111111000000001\n\
        111111111111111111111111111111111111\n";

var maze_table = new Array(18), isvis = new Array(18);
var begin_row = 0, begin_col = 0, target_row = 0, target_col = 0;

// 根据字符串创建maze的图形界面
window.onload = function(){
    var now_td, now_tr, row = 0, col = 0;
    var now_table = $("#maze_table");
    for(var i in maze){
        if(i == 0){
            maze_table[row] = new Array(36);
            isvis[row] = new Array(36);
            now_tr = $('<tr></tr>');
        }
        if(maze.charAt(i) == '\n'){
            row += 1;
            col = 0;
            maze_table[row] = new Array(36);
            isvis[row] = new Array(36);
            now_table.append(now_tr);
            now_tr = $('<tr></tr>');
        }
        else if(maze.charAt(i) == '1'){
            maze_table[row][col] = 1;
            isvis[row][col] = 0;
            now_td = $('<td></td>').attr("style", "background:black;boarder:none;").attr("id", row.toString() + "_" + col.toString())
            now_tr.append(now_td);
            col += 1;
        }
        else if(maze.charAt(i) == '0'){
            maze_table[row][col] = 0;
            isvis[row][col] = 0;
            now_td = $('<td></td>').attr("style", "background:yellow;boarder:none;").attr("id", row.toString() + "_" + col.toString());
            now_tr.append(now_td);
            col += 1;
        }
        else if(maze.charAt(i) == 'S'){
            begin_row = row;
            begin_col = col;
            maze_table[row][col] = 'S';
            isvis[row][col] = 1;
            now_td = $('<td></td>').attr("style", "background:blue;boarder:none;").attr("id", row.toString() + "_" + col.toString());
            now_tr.append(now_td);
            col += 1
        }
        else if(maze.charAt(i) == 'E'){
            target_row = row;
            target_col = col;
            maze_table[row][col] = 'E';
            isvis[row][col] = 0;
            now_td = $('<td></td>').attr("style", "background:green;boarder:none;").attr("id", row.toString() + "_" + col.toString());
            now_tr.append(now_td);
            col += 1;
        }
    }
}

// road保存最终路径，road_queue用于bfs
var road = [], road_queue = [];
var row_steps = [-1, 0, 1, 0], col_steps = [0, 1, 0, -1];

// 结点（保存结点的行、列和父节点）
var Node = function(row, col, father){
    this.row = row;
    this.col = col;
    this.father = father;
}

// 判断下一个结点是否合法
function isVaild(row, col){
    if(row < 0 || row >= 18 || col < 0 || col >= 36)return false;
    if(maze_table[row][col] == 1)return false;
    if(isvis[row][col] == 1)return false;
    return true;
}

// 广搜
function bfs(){
    var begin_node = new Node(begin_row, begin_col, null), flag = 0;
    road_queue.push(begin_node);
    while(road_queue.length > 0 && flag == 0){
        var now_father = road_queue[0];
        var row = now_father.row, col = now_father.col;
        for(var i = 0; i < 4; ++i){
            if(isVaild(row + row_steps[i], col + col_steps[i])){
                var now_node = new Node(row + row_steps[i], col + col_steps[i], now_father)
                road_queue.push(now_node);
                isvis[row + row_steps[i]][col + col_steps[i]] = 1;
                if(now_node.row == target_row && now_node.col == target_col){
                    flag = 1;
                    break;
                }
            }
        }
        road_queue.shift();
    }
    // 如果找到了路径，则将路径保存下来
    if(flag){
        var end_node = road_queue[road_queue.length - 1];
        var p = end_node;
        while(p.father != null){
            road.push(p);
            p = p.father;
        }
        road.push(p);
    }
    else{
        alert("Wrong!");
    }
}

// 模拟走迷宫
function changeColor(now_father, now){
    $('#' + now.row.toString() + '_' + now.col.toString()).attr("style", "background:blue");
    $('#' + now_father.row.toString() + '_' + now_father.col.toString()).attr("style", "background:yellow");
}

// 模拟走迷宫的延时
function delay(){
    return new Promise(resolve => setTimeout(resolve, 100));
}

// 异步解决整个延时
async function delayChangeColor(now_father, now){
    await delay();
    changeColor(now_father, now);
}

// 走迷宫的整个步骤
async function run(){
    bfs();
    var now_father = road[road.length - 1], now;
    for(var i = road.length - 2; i >= 0; --i){
        console.log(now_father.row.toString() + "," + now_father.col.toString())
        now = road[i];
        await delayChangeColor(now_father, now);
        now_father = now;
    }
    alert("Successful!");
    location.reload();
}