module SS(
// input signals
    clk,
    rst_n,
    in_valid,
    matrix,
    matrix_size,
// output signals
    out_valid,
    out_value
);
input               clk, rst_n, in_valid;
input        [15:0] matrix;
input               matrix_size;

output logic        out_valid;
output logic [39:0] out_value;
logic cycle;
//	logic write,next_write;
reg [15:0]W4[1:4][1:4];
reg [15:0]X4[1:4][1:4];
reg [15:0]W2[1:2][1:2];
reg [15:0]X2[1:2][1:2];
reg [15:0]Y2[1:2][1:2];
reg [15:0]Y4[1:4][1:4];
wire [39:0] outp_south0, outp_south1, outp_south2, outp_south3, outp_south4, outp_south5, outp_south6, outp_south7, outp_south8, outp_south9, 	outp_south10, outp_south11, outp_south12, outp_south13, outp_south14, outp_south15;
wire [31:0] outp_east0, outp_east1, outp_east2, outp_east3, outp_east4, outp_east5, outp_east6, outp_east7, outp_east8, outp_east9, outp_east10, 	outp_east11, outp_east12, outp_east13, outp_east14, outp_east15;
wire [63:0] result0, result1, result2, result3, result4, result5, result6, result7, result8, result9, result10, result11, result12, result13, result14, result15;

always@(posedge clk or negedge rst_n) begin
	
	if(rst_n) begin
		out_valid <= 0;
	end else begin
//		cycle 	<= cycle + 1;
//		write        	<= matrix;
//		next_write<= write  ;
		if (matrix_size == 1'b0)begin
			if(cycle == 8) begin
				cycle <= 0;
			end
			else begin
				cycle <= cycle + 1;
			end
		end
		else begin
			if(cycle == 32) begin
				cycle <= 0;
			end
			else begin
				cycle <= cycle + 1;
			end
		end
	end
end
always_comb begin
	if (matrix_size == 1'b0)begin
			case(cycle)
				1:	W2[1][1] = matrix;
				2:	W2[1][2] = matrix;
				3:	W2[2][1] = matrix;
				4:	W2[2][2] = matrix;
				5:	X2[1][1]  = matrix;
				6:	X2[1][2]  = matrix;
				7:	X2[2][1]  = matrix;
				8:	X2[2][2]  = matrix;
			endcase
	end
	if (matrix_size==1'b1)begin
			case(cycle)
				1:	W4[1][1] = matrix;
				2:	W4[1][2] = matrix;
				3:	W4[1][3] = matrix;
  				4:	W4[1][4] = matrix;
 				5:	W4[2][1] = matrix;
 				6:	W4[2][2] = matrix;
  				7:	W4[2][3] = matrix;
 				8:	W4[2][4] = matrix;
				9:	W4[3][1] = matrix;
  				10:	W4[3][2] = matrix;
  				11:	W4[3][3] = matrix;
  				12:	W4[3][4] = matrix;
  				13:	W4[4][1] = matrix;
  				14:	W4[4][2] = matrix;
  				15:	W4[4][3] = matrix;
  				16:	W4[4][4] = matrix;
				17: 	X4[1][1]  = matrix;
				18:	X4[1][2]  = matrix;
				19: 	X4[1][3]  = matrix;
				20: 	X4[1][4]  = matrix;
				21: 	X4[2][1]  = matrix;
				22: 	X4[2][2]  = matrix;
				23:	X4[2][3]  = matrix;
				24: 	X4[2][4]  = matrix;
				25: 	X4[3][1]  = matrix;
				26: 	X4[3][2]  = matrix;
				27: 	X4[3][3]  = matrix;
				28: 	X4[3][4]  = matrix;
				29: 	X4[4][1]  = matrix;
				30: 	X4[4][2]  = matrix;
				31: 	X4[4][3]  = matrix;
				32: 	X4[4][4]  = matrix;
			endcase
	end
end

//from north and west
block P0 (W4[1][1], X4[1][1], clk, rst_n, outp_south0, outp_east0, result0)if(matrix_size == 1'b1);

//from north
block P1 (W4[1][2], outp_east0, clk, rst_n, outp_south1, outp_east1, result1)if(matrix_size == 1'b1);
block P2 (W4[1][3], outp_east1, clk, rst_n, outp_south2, outp_east2, result2)if(matrix_size == 1'b1);
block P3 (W4[1][4], outp_east2, clk, rst_n, outp_south3, outp_east3, result3)if(matrix_size == 1'b1);
	
//from west
block P4 (outp_south0, X4[2][1], clk, rst_n, outp_south4, outp_east4, result4)if(matrix_size == 1'b1);
block P8 (outp_south4, X4[3][1], clk, rst_n, outp_south8, outp_east8, result8)if(matrix_size == 1'b1);
block P12 (outp_south8, X4[4][1], clk, rst_n, outp_south12, outp_east12, result12)if(matrix_size == 1'b1);
	
//no direct inputs
//second row
block P5 (outp_south1, outp_east4, clk, rst_n, outp_south5, outp_east5, result5)if(matrix_size == 1'b1);
block P6 (outp_south2, outp_east5, clk, rst_n, outp_south6, outp_east6, result6)if(matrix_size == 1'b1);
block P7 (outp_south3, outp_east6, clk, rst_n, outp_south7, outp_east7, result7)if(matrix_size == 1'b1);

//third row
block P9 (outp_south5, outp_east8, clk, rst_n, outp_south9, outp_east9, result9)if(matrix_size == 1'b1);
block P10 (outp_south6, outp_east9, clk, rst_n, outp_south10, outp_east10, result10)if(matrix_size == 1'b1);
block P11 (outp_south7, outp_east10, clk, rst_n, outp_south11, outp_east11, result11)if(matrix_size == 1'b1);

//fourth row
block P13 (outp_south9, outp_east12, clk, rst_n, outp_south13, outp_east13, result13)if(matrix_size == 1'b1);
block P14 (outp_south10, outp_east13, clk, rst_n, outp_south14, outp_east14, result14)if(matrix_size == 1'b1);
block P15 (outp_south11, outp_east14, clk, rst_n, outp_south15, outp_east15, result15)if(matrix_size == 1'b1);

// First block
block P0 (inp_north0, inp_west0, clk, rst_n, outp_south0, outp_east0, result0)if(matrix_size == 1'b0);
block P1 (inp_north1, outp_east0, clk, rst_n, outp_south1, outp_east1, result1)if(matrix_size == 1'b0);

// Second block
block P2 (outp_south0, inp_west2, clk, rst_n, outp_south2, outp_east2, result2)if(matrix_size == 1'b0);
block P3 (outp_south1, outp_east2, clk, rst_n, outp_south3, outp_east3, result3)if(matrix_size == 1'b0);

always @(posedge rst or posedge clk) begin
	if(matrix_size == 1'b0) begin
		case(cycle)
			1:out_value <= 
endmodule

module block(inp_north, inp_west, clk, rst, outp_south, outp_east, result);
	input [31:0] inp_north, inp_west;
	output reg [31:0] outp_south, outp_east;
	input clk, rst;
	output reg [63:0] result;
	wire [63:0] multi;
	always @(posedge rst or posedge clk) begin
		if(rst) begin
			result <= 0;
			outp_east <= 0;
			outp_south <= 0;
		end
		else begin
			result <= result + multi;
			outp_east <= inp_west;
			outp_south <= inp_north;
		end
	end
	assign multi = inp_north*inp_west;
endmodule