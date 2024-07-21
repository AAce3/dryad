use pyrrhic_rs::{EngineAdapter, TBError, TableBases, WdlProbeResult};

use super::{
    atks,
    bitboard::{self, Bitboard},
    types::{square, Color, Piece, Square},
    zobrist::{self, halfmove_zobrist, Zobrist},
};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Board {
    piece_bbs: [Bitboard; 6],
    colors: [Bitboard; 2],
    active_color: Color,
    passant_square: Option<Square>,
    halfmove_clock: u8,
    zobrist: Zobrist,
    castling_rights: [bool; 4],
}

#[derive(Clone, Copy)]
pub enum Castling {
    WK,
    WQ,
    BK,
    BQ,
}

impl Board {
    pub fn new() -> Self {
        let mut default = Board::default();
        default
            .parse_fen(&String::from(
                "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
            ))
            .unwrap();
        default
    }

    pub fn is_material_draw(&self) -> bool {
        // material draw
        let can_force_mate = self.piece_bbs[Piece::P as usize] > 0
            || self.piece_bbs[Piece::R as usize] > 0
            || self.piece_bbs[Piece::Q as usize] > 0
            || self.piece_bb(Piece::B, Color::White).count_ones() >= 2
            || self.piece_bb(Piece::B, Color::Black).count_ones() >= 2
            || (self.piece_bb(Piece::B, Color::White).count_ones() >= 1
                && self.piece_bb(Piece::N, Color::White) >= 1)
            || (self.piece_bb(Piece::B, Color::Black).count_ones() >= 1
                && self.piece_bb(Piece::N, Color::Black) >= 1);

        !can_force_mate || self.halfmove_clock() >= 100
    }

    #[inline]
    pub fn is_color(&self, square: Square, color: Color) -> bool {
        bitboard::is_set(self.colors[color as usize], square)
    }

    #[inline]
    pub fn get_piece(&self, square: Square) -> Piece {
        let mut result = 0;
        for i in 0..6 {
            result += (bitboard::is_set(self.pieces()[i as usize], square) as u8) * (i + 1);
        }
        Piece::from(result.wrapping_sub(1))
    }

    #[inline]
    pub fn color_bb(&self, color: Color) -> Bitboard {
        self.colors[color as usize]
    }

    #[inline]
    pub fn piece_bb(&self, piece: Piece, color: Color) -> Bitboard {
        self.piece_bbs[piece as usize] & self.colors[color as usize]
    }

    #[inline]
    pub fn occupancy(&self) -> Bitboard {
        self.colors[Color::White as usize] | self.colors[Color::Black as usize]
    }

    #[inline]
    pub fn diagonal_sliders(&self, color: Color) -> Bitboard {
        (self.piece_bbs[Piece::B as usize] | self.piece_bbs[Piece::Q as usize])
            & self.colors[color as usize]
    }

    #[inline]
    pub fn orthogonal_sliders(&self, color: Color) -> Bitboard {
        (self.piece_bbs[Piece::R as usize] | self.piece_bbs[Piece::Q as usize])
            & self.colors[color as usize]
    }

    #[inline]
    pub fn pieces(&self) -> &[Bitboard; 6] {
        &self.piece_bbs
    }

    #[inline]
    pub fn halfmove_clock(&self) -> u8 {
        self.halfmove_clock
    }

    #[inline]
    pub(super) fn passant_square(&self) -> Option<Square> {
        self.passant_square
    }

    #[inline]
    pub fn active_color(&self) -> Color {
        self.active_color
    }

    #[inline]
    pub fn castling(&self, castling: Castling) -> &bool {
        &self.castling_rights[castling as usize]
    }

    #[inline]
    pub(super) fn castling_mut(&mut self, castling: Castling) -> &mut bool {
        &mut self.castling_rights[castling as usize]
    }

    #[inline]
    pub fn get_hash(&self) -> Zobrist {
        self.zobrist
    }

    #[inline]
    pub fn get_halfmove_hash(&self) -> Zobrist {
        self.get_hash() ^ halfmove_zobrist(self.halfmove_clock)
    }


    #[inline]
    pub(super) fn zobrist_mut(&mut self) -> &mut Zobrist {
        &mut self.zobrist
    }

    #[inline]
    pub fn piecetype(&self, piecetype: Piece) -> Bitboard {
        self.piece_bbs[piecetype as usize]
    }

    #[inline]
    pub fn is_kp(&self) -> bool {
        self.piecetype(Piece::N) == 0
            && self.piecetype(Piece::B) == 0
            && self.piecetype(Piece::R) == 0
            && self.piecetype(Piece::Q) == 0
    }
}

// these methods all involve changing the zobrist hash.
impl Board {
    #[inline]
    pub(super) fn add_piece(&mut self, square: Square, piece: Piece, color: Color) {
        bitboard::set_bit(&mut self.piece_bbs[piece as usize], square);
        bitboard::set_bit(&mut self.colors[color as usize], square);

        *self.zobrist_mut() ^= zobrist::psqt_zobrist(piece, square, color);
    }

    #[inline]
    pub(super) fn remove_piece(&mut self, square: Square, piece: Piece, color: Color) {
        bitboard::clear_bit(&mut self.piece_bbs[piece as usize], square);
        bitboard::clear_bit(&mut self.colors[color as usize], square);

        *self.zobrist_mut() ^= zobrist::psqt_zobrist(piece, square, color);
    }

    #[inline]
    pub(super) fn move_piece(&mut self, from: Square, to: Square, piece: Piece, color: Color) {
        self.remove_piece(from, piece, color);
        self.add_piece(to, piece, color)
    }

    #[inline]
    pub(super) fn swap_sides(&mut self) {
        self.active_color = !self.active_color;
        *self.zobrist_mut() ^= zobrist::turn_zobrist();
    }

    #[inline]
    pub(super) fn set_castling(&mut self, castling: Castling, value: bool) {
        let current_castling = self.castling_rights[castling as usize];
        *self.castling_mut(castling) = value;

        *self.zobrist_mut() ^= zobrist::castling_zobrist(current_castling, castling);
        *self.zobrist_mut() ^= zobrist::castling_zobrist(value, castling);
    }

    #[inline]
    pub(super) fn set_fifty(&mut self, value: u8) {
        self.halfmove_clock = value;
    }

    #[inline]
    pub(super) fn reset_fifty(&mut self) {
        self.set_fifty(0)
    }

    #[inline]
    pub(super) fn increment_fifty(&mut self) {
        self.set_fifty(self.halfmove_clock() + 1)
    }

    #[inline]
    fn current_ep_zob(&self) -> Zobrist {
        match self.passant_square() {
            Some(square) => zobrist::passant_zobrist(square),
            None => 0,
        }
    }

    #[inline]
    pub(super) fn set_ep(&mut self, square: Square) {
        *self.zobrist_mut() ^= self.current_ep_zob();
        *self.zobrist_mut() ^= zobrist::passant_zobrist(square);
        self.passant_square = Some(square)
    }

    #[inline]
    pub(super) fn reset_passant(&mut self) {
        *self.zobrist_mut() ^= self.current_ep_zob();
        self.passant_square = None
    }

    #[inline]
    pub(super) fn update_castle(&mut self) {
        let white_rooks = self.piece_bb(Piece::R, Color::White);
        let black_rooks = self.piece_bb(Piece::R, Color::Black);

        let wk_rook = bitboard::is_set(white_rooks, square::H1 as Square);
        let wq_rook = bitboard::is_set(white_rooks, square::A1 as Square);

        let bk_rook = bitboard::is_set(black_rooks, square::H8 as Square);
        let bq_rook = bitboard::is_set(black_rooks, square::A8 as Square);

        let white_king = self.piece_bb(Piece::K, Color::White);
        let black_king = self.piece_bb(Piece::K, Color::Black);

        let w_king = bitboard::is_set(white_king, square::E1 as Square);
        let b_king = bitboard::is_set(black_king, square::E8 as Square);

        self.set_castling(
            Castling::WK,
            *self.castling(Castling::WK) && w_king && wk_rook,
        );
        self.set_castling(
            Castling::WQ,
            *self.castling(Castling::WQ) && w_king && wq_rook,
        );
        self.set_castling(
            Castling::BK,
            *self.castling(Castling::BK) && b_king && bk_rook,
        );
        self.set_castling(
            Castling::BQ,
            *self.castling(Castling::BQ) && b_king && bq_rook,
        );
    }
}

impl Board {
    pub fn probe_wdl(&self, tablebase: TableBases<Self>) -> Result<WdlProbeResult, TBError> {
        tablebase.probe_wdl(
            self.color_bb(Color::White),
            self.color_bb(Color::White),
            self.piecetype(Piece::K),
            self.piecetype(Piece::Q),
            self.piecetype(Piece::R),
            self.piecetype(Piece::B),
            self.piecetype(Piece::N),
            self.piecetype(Piece::P),
            self.passant_square().unwrap_or(0) as u32,
            self.active_color() == Color::White,
        )
    }
}

// Syzygy tablebases interface via Pyrrhic
impl EngineAdapter for Board {
    fn pawn_attacks(color: pyrrhic_rs::Color, square: u64) -> u64 {
        let bitboard = 1 << square;
        let forward_bb = bitboard::forward(
            bitboard,
            match color {
                pyrrhic_rs::Color::Black => Color::Black,
                pyrrhic_rs::Color::White => Color::White,
            },
        );
        bitboard::shift(forward_bb, bitboard::Direction::E)
            | bitboard::shift(forward_bb, bitboard::Direction::W)
    }

    fn knight_attacks(square: u64) -> u64 {
        atks::knight_attacks(square as u8)
    }

    fn bishop_attacks(square: u64, occupied: u64) -> u64 {
        atks::bishop_attacks(square as u8, occupied)
    }

    fn rook_attacks(square: u64, occupied: u64) -> u64 {
        atks::rook_attacks(square as u8, occupied)
    }

    fn queen_attacks(square: u64, occupied: u64) -> u64 {
        atks::rook_attacks(square as u8, occupied) | atks::bishop_attacks(square as u8, occupied)
    }

    fn king_attacks(square: u64) -> u64 {
        atks::king_attacks(square as u8)
    }
}
